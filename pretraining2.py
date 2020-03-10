
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D,AveragePooling2D

from sklearn.model_selection import train_test_split
import cv2

#keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics

train = pd.read_csv("../Downloads/emnist-train-letters.csv",delimiter = ',')
testt = pd.read_csv("../Downloads/emnist-test-letters.csv", delimiter = ',')
mapp = pd.read_csv("../Downloads/emnist-letters-mapping.txt", delimiter = ' ', \
                   index_col=0, header=None, squeeze=True)

HEIGHT = 28
WIDTH = 28

train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
del train

test_x = testt.iloc[:,1:]
test_y = testt.iloc[:,0]
del testt

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
print ("train_x:",train_x.shape)

test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)
print ("test_x:",test_x.shape)

# Normalise
train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255



num_classes = train_y.nunique()
print(num_classes)

train_y = train_y - 1
test_y = test_y - 1
train_y  = np_utils.to_categorical(train_y,num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size= 0.10, random_state=7)
input_shape = (28,28,1)

feature_layers = [Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)),
AveragePooling2D(pool_size=(2, 2)),
Flatten()]
classification_layers =[Dense(128, activation=tf.nn.relu),
Dense(num_classes,activation=tf.nn.softmax)]

model = Sequential(feature_layers + classification_layers)

"""
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.5))
model.add(Dense(num_classes,activation=tf.nn.softmax))



model = Sequential()
model.add(Dense(128, input_shape=(28,28,1)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(64, input_shape=(28,28,128)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(26,input_shape=(28,28,64)))
print(num_classes)"""

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    train_x, train_y,
    epochs=50,
    batch_size=512,
    verbose=1,
    validation_data=(val_x, val_y))


model.save('pre-trained_model1.h5')

