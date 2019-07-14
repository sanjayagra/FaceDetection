
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
input_shape = (28, 28, 3)


# In[2]:


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})
np.random.seed(2017)


# In[3]:


def define_model():
    input_1 = Input(shape=input_shape, name='image')
    convolve = Conv2D(64, kernel_size=(3, 3), padding='same')(input_1)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(128, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(256, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = Conv2D(256, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Conv2D(512, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = Conv2D(512, kernel_size=(3, 3), padding='same')(convolve)
    convolve = BatchNormalization()(convolve)
    convolve = Activation('swish')(convolve)
    convolve = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(convolve)
    convolve = Flatten()(convolve)
    concat = Dense(512, activation='swish', kernel_initializer='he_normal')(convolve)
    concat = Dropout(0.3)(concat)
    concat = Dense(256, activation='swish', kernel_initializer='he_normal')(concat)
    concat = Dropout(0.3)(concat)
    predict = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(concat)
    model = Model(inputs=input_1, output=predict)
    optimizer = Adam(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# In[4]:


model = define_model()
model.summary()


# In[13]:


params = {}
params['horizontal_flip'] = True
params['vertical_flip'] = True
params['zoom_range'] = 0.2
params['rotation_range'] = 10


# In[14]:


generator = ImageDataGenerator(**params)
def dataflow(image, label):
    flow_1 = generator.flow(image, label, batch_size=32,seed=2017)
    while True:
        tuple_1 = flow_1.next()
        yield tuple_1[0], tuple_1[1]


# In[15]:


def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=25, mode="min")
    path =  path_cb + 'data/model/model_1/model_{}.hdf5'.format(suffix)
    save = ModelCheckpoint(path, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(path_cb + 'data/model/model_1/logger_{}.log'.format(suffix))
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
    return [stop, save, reduce, logger]


# In[16]:


train_image = np.load(path_cb + 'data/train/x_train_1.npy')
train_label = np.load(path_cb + 'data/train/y_train_1.npy')
test_image = np.load(path_cb + 'data/score/x_test_1.npy')
test_label = np.load(path_cb + 'data/score/y_test_1.npy')
train_generator = dataflow(train_image, train_label)
test_generator = (test_image, test_label)


# In[ ]:


params = {}
params['generator'] = train_generator
params['validation_data'] = test_generator
params['steps_per_epoch'] = 20
params['epochs'] = 5
params['verbose'] = 1
params['callbacks'] = callbacks(2)
model_1 = define_model()
model_1.fit_generator(**params)
K.clear_session()

