from tensorflow import keras

from keras.models import load_model
from keras.utils import load_img, img_to_array
# from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

import matplotlib.pyplot as plt
import cv2

model=load_model('model_saved.h5')
 
image = load_img('dataset/test/royce.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - EV , 1- nonEV): ", label[0][0])

if label<1:
    print('The car is Electrical Vehicle')
    
else:
    print('The car is not Electrical Vehicle')
   



# # create a batch of size 1 [N,H,W,C]
# img = np.expand_dims(img, axis=0)
# prediction = model.predict(img, batch_size=None,steps=1) #gives all class prob.
# if(prediction[:,:]>0.5):
#     value ='EV :%1.2f'%(prediction[0,0])
#     plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
# else:
#     value ='nonEV :%1.2f'%(1.0-prediction[0,0])
#     plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

# plt.imshow(image)
# plt.show()