import keras
import numpy as np
from keras.applications import vgg19,resnet50,inception_v3

vgg_19 = vgg19.VGG19(weights='imagenet')
resnet = resnet50.ResNet50(weights='imagenet')
inception = inception_v3.InceptionV3(weights='imagenet')