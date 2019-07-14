
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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


# In[3]:


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})
np.random.seed(2017)


# In[4]:


def define_model(rate):
    # inputs
    input_1 = Input(shape=input_shape, name='image')
    vgg = VGG16(input_tensor=input_1, pooling='max', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    convolve = Dropout(0.3)(vgg.output)
    concat = Dense(512, activation='swish', kernel_initializer='he_normal')(convolve)
    concat = Dropout(0.2)(concat)
    concat = Dense(128, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.2)(concat)
    concat = Dense(36, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.2)(concat)
    predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
    # model
    model = Model(inputs=input_1, output=predict)
    optimizer = Adam(lr=rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[5]:


params = {}
params['horizontal_flip'] = True
params['vertical_flip'] = True
params['zoom_range'] = 0
params['rotation_range'] = 0
params['width_shift_range'] = 0
params['height_shift_range'] = 0


# In[6]:


generator = ImageDataGenerator(**params)


# In[7]:


def dataflow(image, label):
    flow_1 = generator.flow(image, label, batch_size=32,seed=2017)
    while True:
        tuple_1 = flow_1.next()
        yield tuple_1[0], tuple_1[1]


# In[11]:


def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=25, mode="min")
    path =  path_cb + 'data/model/model_1/model_{}.hdf5'.format(suffix)
    save = ModelCheckpoint(path, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(path_cb + 'data/model/model_1/logger_{}.log'.format(suffix))
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
    return [stop, save, reduce, logger]


# In[12]:


train_image = np.load(path_cb + 'data/train/x_train_1.npy')
train_label = np.load(path_cb + 'data/train/y_train_1.npy')
test_image = np.load(path_cb + 'data/score/x_test_1.npy')
test_label = np.load(path_cb + 'data/score/y_test_1.npy')
train_generator = dataflow(train_image, train_label)
test_generator = (test_image, test_label)


# In[14]:


params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 10
params['epochs'] = 5
params['verbose'] = 1
params['callbacks'] = callbacks(4)
model_1 = define_model(1e-4)
model_1.fit_generator(**params)
model_1.load_weights(path_cb + 'data/model/model_1/model_1.hdf5')

