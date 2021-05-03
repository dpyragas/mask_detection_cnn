#!/usr/bin/env python
# coding: utf-8

#Importing required packages


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from tensorflow import keras
from tensorflow.keras import layers, models,regularizers
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image





batch_size = 64


#Defining the train dataset and doing data augmentation on training dataset


train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   vertical_flip =  True , 
                                   rotation_range=40,
                                   brightness_range = (0.5, 1.5),
                                   horizontal_flip = True)
train_set = train_datagen.flow_from_directory(
        'data/Train',
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle= True,
        class_mode='binary')


#Defining the test dataset


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'data/Test',
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')


#Defining the validation dataset


val_datagen = ImageDataGenerator(rescale=1./255)
val_set = test_datagen.flow_from_directory(
        'data/Validation',
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')


#Checking for classes, later we will use them for unknown images


classes = {0:"with_mask",1:"without_mask"}
for i in classes.items():
    print(i)




#How many images of each label each dataset has

print('Training images without mask:', len(os.listdir('data/Train/WithoutMask')))
print('Training images with mask:', len(os.listdir('data/Train/WithMask')))
print('Testing images without mask:', len(os.listdir('data/Test/WithoutMask')))
print('Testing images with mask:', len(os.listdir('data/Test/WithMask')))
print('Validation images without mask:', len(os.listdir('data/Validation/WithoutMask')))
print('Validation images with mask:', len(os.listdir('data/Validation/WithMask')))


'''BUILDING THE MODEL'''


model = models.Sequential()
#The first CNN layer followed by Relu and MaxPooling layers and BatchNormalization
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
#The second CNN layer followed by Relu and MaxPooling layers and BatchNormalization
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
#The third CNN layer followed by Relu and MaxPooling layers and BatchNormalization
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
#The fourth CNN layer followed by Relu and MaxPooling layers and BatchNormalization
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Flatten())
#Dense layer of 512 neurons
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
#The Final layer with output as 1, as it has only two possible outputs - 0 or 1.
model.add(Dense(1, activation='sigmoid'))
model.summary()




total_sample=train_set.n


#Compiling the model


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


#Running CNN

history=model.fit(train_set,
    validation_data = val_set,
    steps_per_epoch=int(total_sample/batch_size),
    epochs = 30,
    shuffle=True)


#Model evaluation with accuracy and loss

model.evaluate(test_set)




y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred,axis=1)


#Plotting the loss and accuracy on validation and train datasets


sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)



#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#We can input the images now and check for predictions.

test_image = image.load_img('data/Test/WithMask/3.png', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
train_set.class_indices
if result[0][0] == 0:
  prediction = 'with_mask'
else:
  prediction = 'without_mask'
print(prediction)



