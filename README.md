# Tiny-ImageNet-Pretrained-Model
## This project is for research usage. 

These pretrained models can be used in transfer learning, model distillation or model extraction attacks.



I trained these models from scratch. The accuracy on val test is about 51%~54%, depending different models.

Pretrained Models can be found from [here](https://drive.google.com/drive/folders/1qV3bZAhJCC23qNZeTS0Meo-wt6RW2s4B?usp=sharing)

To train your model, you can

``python pretrain_models.py --gpu [id] --arch [resnet, wrn, vgg, mobilenet] --path [path to dataset]
``
