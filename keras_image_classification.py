import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg19,resnet50,inception_v3

# load model and set weights
vgg_19 = vgg19.VGG19(weights='imagenet')
resnet = resnet50.ResNet50(weights='imagenet')
#inception = inception_v3.InceptionV3(weights='imagenet')

# load image and set desired size then convert to numpy array since it has width height and channel
# load_img() loads an image in PIL format with width and height dimensions
img = "sample_images/test.jpeg"
loaded_img = img_to_array(load_img(img, target_size=(224,224)))
# convert image to 4D tensor. the extra dimension is batch-size
input = np.expand_dims(loaded_img,axis=0)

#preprocess image accordig to the various models

vgg_preprocess = vgg19.preprocess_input(input.copy())
resnet_preprocess = resnet50.preprocess_input(input.copy())
#inception_preprocess = inception_v3.preprocess_input(input.copy())
# make predictions
predictions_vgg19 = vgg_19.predict(vgg_preprocess)
label_vgg19 = decode_predictions(predictions_vgg19)
print (f'Vgg_19 prediction={label_vgg19[0][0][1]}')

predictions_resnet = resnet.predict(resnet_preprocess)
label_resnet = decode_predictions(predictions_resnet)
print (f'Resnent prediction={label_resnet[0][0][1]}')

# predictions_inception = inception.predict(inception_preprocess)
# label_inception = decode_predictions(predictions_inception)
# print (f'Inception prediction={label_inception}')