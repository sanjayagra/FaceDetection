
# coding: utf-8

# In[2]:


import pandas as pd, os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras
import os, time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
path = '/home/ec2-user/Sanjay/projects/6_FaceDetection/Data/'
path_cb = '/home/ec2-user/Sanjay/projects/6_FaceDetection/1_gender/'
input_shape = (224, 224, 3)


# In[4]:


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})
np.random.seed(2017)


# In[5]:


def define_model():
    # inputs
    input_1 = Input(shape=input_shape, name='image')
    vgg = VGG16(input_tensor=input_1, pooling='max', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    convolve = Dropout(0.3)(vgg.output)
    concat = Dense(256, activation='relu')(convolve)
    concat = Dropout(0.5)(concat)
    concat = Dense(64, activation='relu')(concat)
    concat = Dropout(0.3)(concat)
    concat = Dense(16, activation='relu')(concat)
    concat = Dropout(0.1)(concat)
    predict = Dense(1, activation='sigmoid')(concat)
    # model
    model = Model(inputs=input_1, output=predict)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    return model


# In[6]:


def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=25, mode="min")
    path =  path_cb + 'data/model/model_1/model_{}.hdf5'.format(suffix)
    save = ModelCheckpoint(path, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(path_cb + 'data/model/model_1/logger_{}.log'.format(suffix))
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
    return [stop, save, reduce, logger]


# In[3]:


train_image = np.load(path_cb + 'data/train/x_train_1_all_cropped.npy')
train_label = np.load(path_cb + 'data/train/y_train_1_all_cropped.npy')
test_image = np.load(path_cb + 'data/score/x_test_1_all_cropped.npy')
test_label = np.load(path_cb + 'data/score/y_test_1_all_cropped.npy')


# In[7]:


import time
start_time = time.time()
model = define_model()
model.fit(train_image, train_label,
          batch_size=128,
          epochs=100,
          verbose=1,
          callbacks = callbacks(3), 
          validation_data=(test_image, test_label))
print (time.time() - start_time)

