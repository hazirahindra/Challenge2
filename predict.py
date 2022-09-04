from tensorflow import keras

from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import cv2

model=load_model('model_saved.h5')
 
image = load_img('dataset/test/rsz_swift.png', target_size=(224, 224))

img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)

if label<0.01:
    print("This car is an Electric Car!")
else:
    print ("This car is not an Electric Car!")