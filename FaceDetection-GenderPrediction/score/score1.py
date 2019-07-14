
# coding: utf-8

# In[2]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
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


import keras
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', activation ='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(16, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])


# In[5]:


def callbacks(suffix):
    stop = EarlyStopping('val_loss', patience=25, mode="min")
    path =  path_cb + 'data/model/model_1/model_{}.hdf5'.format(suffix)
    save = ModelCheckpoint(path, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(path_cb + 'data/model/model_1/logger_{}.log'.format(suffix))
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
    return [stop, save, reduce, logger]


# In[6]:


train_image = np.load(path_cb + 'data/train/x_train_1_all_cropped.npy')
train_label = np.load(path_cb + 'data/train/y_train_1_all_cropped.npy')
test_image = np.load(path_cb + 'data/score/x_test_1_all_cropped.npy')
test_label = np.load(path_cb + 'data/score/y_test_1_all_cropped.npy')


# In[7]:


yage_train = np.load(path_cb + 'data/train/yage_train_1_all.npy')
yage_test = np.load(path_cb + 'data/train/yage_test_1_all.npy')
yrace_train = np.load(path_cb + 'data/train/yrace_train_1_all.npy')
yrace_test = np.load(path_cb + 'data/train/yrace_test_1_all.npy')


# In[8]:


model.load_weights(path_cb + 'data/model/model_1/model_{}.hdf5'.format(1))
pred = pd.DataFrame(model.predict(test_image))
pred.columns = ['score']
pred['act'] = test_label
pred['age'] = yage_test
pred['race'] = yrace_test


# In[9]:


threshold=0.5
pred['pred_class'] = pred['score'].map(lambda x : 1 if x>threshold else 0)
pred['check'] = pred.apply(lambda x : 1 if x['pred_class']==x['act'] else 0, axis=1)


# In[10]:


pred['check'].sum()/pred['check'].count()


# In[15]:


age_grp = []
for index, row in pred.iterrows():
    if (row['age']>=0) and (row['age']<=2):
        age_grp.append('0-2 Years')
    elif (row['age']>2) and (row['age']<=5):
        age_grp.append('3-5 Years')
    elif (row['age']>5) and (row['age']<=10):
        age_grp.append('6-10 Years')
    elif (row['age']>10) and (row['age']<=20):
        age_grp.append('11-20 Years')
    elif (row['age']>20) and (row['age']<=50):
        age_grp.append('21-50 Years')
    elif (row['age']>50) and (row['age']<=80):
        age_grp.append('51-80 Years')
    else:
        age_grp.append('80+ Years')
pred['age_grp'] = age_grp


# In[16]:


pred.to_csv(path_cb + 'pred_gender.csv', index=False)


# In[17]:


age_grp_sum = pred.groupby(['age_grp'])['check'].sum().reset_index()
age_grp_count = pred.groupby(['age_grp'])['check'].count().reset_index()
merge = age_grp_sum.merge(age_grp_count, on='age_grp', how='inner')


# In[18]:


merge['accuracy'] = merge.apply(lambda x : x['check_x']/x['check_y'], axis=1)
merge


# In[21]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
imgplot = plt.imshow(test_image[1])
plt.show()

