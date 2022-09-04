#Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# from livelossplot import PlotLossesKeras

#Initialize the image size 
img_width, img_height = 224, 224

#Set the path for train and validation dataset 
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'
nb_train_samples =600        # total no of train samples
nb_validation_samples = 300   # total no of validation samples 
epochs = 20                   # no of epochs 
batch_size = 16               # no of batch size 

# Checking format of Image 
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Image Data Generator 
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# model network 
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# monitor_val_acc = EarlyStopping(monitor = 'val_loss', patience = 5)

# callbacks=[PlotLossesKeras()] # this is a single magic line of code which draw #live chart
 
# model training 
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    # callbacks=[PlotLossesKeras(), monitor_val_acc],
    # verbose=0)

# saving the model training 
model.save('model_saved.h5')
print("\nModel Successfully Saved")

# # get the metrics from history

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc)) 

# # plot accuracy with matplotlib
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Accuracy in training and validation')
# plt.figure()
# plt.show()

# # plot loss with matplotlib
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Loss in training and validation')
# plt.figure()
# plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Model Accuracy
x1 = model.evaluate(train_generator)
x2 = model.evaluate(validation_generator)

print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f'%(x1[1]*100,x1[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f'%(x2[1]*100,x2[0]))