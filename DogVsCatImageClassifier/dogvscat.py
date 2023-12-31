# -*- coding: utf-8 -*-
"""DogVsCat

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WfVdgShlNKVAUiV8RUHFbTMfj7g6UlqO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

!unzip dogs-vs-cats.zip

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

train_ds=keras.utils.image_dataset_from_directory(
    directory="/content/train",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256,256)
)

validation_ds=keras.utils.image_dataset_from_directory(
    directory="/content/test",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256,256)
)

type(train_ds)

##normalizing dataset
def process(image,label):
  image=tf.cast(image/255,tf.float32)
  return image,label

train_ds=train_ds.map(process)
validation_ds=validation_ds.map(process)

validation_ds

## model
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_ds,epochs=10,validation_data=validation_ds)

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import cv2
testimage = cv2.imread('/content/sample_data/beagle-hound-dog.webp')
plt.imshow(testimage)

print(testimage.shape)
testimage = cv2.resize(testimage,(256,256)) #Resizing to 256x256
testimage = testimage.reshape((1,256,256,3))

testimage.shape

output = model.predict(testimage)

output

if output>0.5:
    print("It's a dog!")
else:
    print("It's a cat!")

"""save model"""

model.save("cstvsdat_cnn_classification.h5")

path="/content/drive/MyDrive/Deep Learning/Models/catvsdat_cnn_classification.h5"
model.save(path)

"""Working with saved models"""

import cv2
testimage2 = cv2.imread('/content/sample_data/cat2.jpeg')
plt.imshow(testimage2)
print(testimage2.shape)
testimage2 = cv2.resize(testimage2,(256,256)) #Resizing to 256x256
testimage2 = testimage2.reshape((1,256,256,3))

catvsdogmodel=keras.models.load_model("/content/drive/MyDrive/Deep Learning/Models/catvsdat_cnn_classification.h5")

output = catvsdogmodel.predict(testimage2)
if output>0.5:
    print("It's a dog!")
else:
    print("It's a cat!")

"""converting to tflite"""

tf_lite_converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=tf_lite_converter.convert()

TF_LITE_MODEL_FILE_NAME="catvsdat_cnn_classification.tflite"
open("/content/drive/MyDrive/Deep Learning/Models/catvsdat_cnn_classificationV2.tflite","wb").write(tflite_model)

