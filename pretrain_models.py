import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import resnet
from torch import nn
from tqdm.auto import tqdm
import time
from datetime import timedelta
import torchattacks
import os
import numpy as np
import argparse
import dataloader
import wrn
import vgg
import mobilenetv2
import attacks
from torch.utils.data import TensorDataset, DataLoader
from logging import getLogger
import logging
import random
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
import torchvision.transforms.functional as TF
class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""
def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
					help='model architecture')
parser.add_argument('--dataset', default='tiny', type=str,
					help='which dataset used to train')
parser.add_argument('--path', default='./', type=str,
					help='path to your dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')
parser.add_argument('--seed', default=0, type=int,
					help='seed for initializing training. ')
parser.add_argument('--class_num', default=200, type=int,
					help='num of classes')
parser.add_argument('--save', default='MN.pkl', type=str,
					help='model save name')
parser.add_argument('--exp', default='exp_test_grad', type=str,
					help='exp name')

parser.add_argument('--aug', default=1, type=int,
					help='whether use data augmentation')
parser.add_argument('--gamma', type=float, default=0.1,
					help='LR is multiplied by gamma on schedule.')


args = parser.parse_args()
logger = getLogger()
if not os.path.exists('./pretrained/'+args.dataset+'/'+args.exp):
	os.makedirs('./pretrained/'+args.dataset+'/'+args.exp)
logger = create_logger(
	os.path.join('./pretrained/' + args.dataset + '/' + args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = './pretrained/' + args.dataset + '/' + args.exp + '/' +  args.save
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
setup_seed(args.seed)
wd=args.wd
learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size
torch.backends.cudnn.benchmark = True

if args.arch == 'resnet':
	n = resnet.resnet18(args.dataset).cuda()
elif args.arch == 'wrn':
	n = wrn.WideResNet(num_classes=200).cuda()
elif args.arch == 'vgg':
	n = vgg.vgg19_bn(200).cuda()
elif args.arch == 'mobilenet':
	n = mobilenetv2.mobilenetv2(200).cuda()

transform_test=transforms.Compose([torchvision.transforms.Resize((64,64)),
									   transforms.ToTensor(),
									   ])
dl = dataloader.Data(args.dataset, args.path)
trainloader, testloader = dl.data_loader(transform_test, transform_test, batch_size)
def data_aug(image):
	image = TF.center_crop(image, [int(64.0 * random.uniform(0.95, 1.0)), int(64.0 * random.uniform(0.95, 1.0))])
	image = TF.resize(image, [64, 64])
	noise = torch.randn_like(image).cuda() * 0.001
	image = torch.clamp(image + noise, 0.0, 1.0)
	if random.uniform(0, 1) > 0.5:
		image = TF.vflip(image)
	if random.uniform(0, 1) > 0.5:
		image = TF.hflip(image)
	angles=[-15, 0, 15]
	angle = random.choice(angles)
	image = TF.rotate(image, angle)
	return image

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

Loss = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(n.parameters() ,momentum=args.momentum,
							lr=learning_rate,weight_decay=wd)#+ optimize_parameters
milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)
best_eval_acc = 0.0

for epoch in range(epochs):
	loadertrain = tqdm(trainloader, desc='{} E{:03d}'.format('train', epoch), ncols=0, total=len(trainloader))
	epoch_loss = 0.0
	clean_acc = 0.0
	adv_acc = 0.0
	total=0.0
	for x_train, y_train in loadertrain:
		x_train, y_train = x_train.cuda(), y_train.cuda()
		if args.aug == 1:
			x_train = data_aug(x_train)
		n.train()
		inputs, targets_a, targets_b, lam = mixup_data(x_train, y_train,
													   1.0, True)
		inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs,
													  targets_a, targets_b))
		outputs = n(inputs)
		loss = mixup_criterion(Loss, outputs, targets_a, targets_b, lam)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		clean_acc  += predicted.eq(y_train.data).cuda().sum()
		total += y_train.size(0)
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(loss.data.item()),
								clean_acc=fmt(clean_acc.item() / total * 100))
	scheduler.step()
	if (epoch) % 1 == 0:
		test_loss_cl = 0.0
		test_loss_adv = 0.0
		correct_cl = 0.0
		correct_adv = 0.0
		total = 0.0
		n.eval()
		loadertest = tqdm(testloader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
		with torch.enable_grad():
			for x_test, y_test in loadertest:
				x_test, y_test = x_test.cuda(), y_test.cuda()
				n.eval()
				y_pre = n(x_test)
				loss_cl = Loss(y_pre, y_test)
				test_loss_cl += loss_cl.data.item()
				_, predicted = torch.max(y_pre.data, 1)
				total += y_test.size(0)
				correct_cl += predicted.eq(y_test.data).cuda().sum()
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss_cl=fmt(loss_cl.data.item()),
									   acc_cl=fmt(correct_cl.item() / total * 100))
		if correct_cl.item() / total * 100 > best_eval_acc:
			best_eval_acc = correct_cl.item() / total * 100
			checkpoint = {
				'state_dict': n.state_dict(),
				'epoch': epoch
			}
			torch.save(checkpoint, args.save)



