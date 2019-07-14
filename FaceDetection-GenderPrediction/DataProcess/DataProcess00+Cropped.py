
# coding: utf-8

# In[2]:


import pandas as pd, os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
path = '/home/ec2-user/Sanjay/projects/6_FaceDetection/Data/'
path_cb = '/home/ec2-user/Sanjay/projects/6_FaceDetection/1_gender/'
import cv2
import cvlib as cv
input_shape = (224, 224)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[3]:


images = []
img_path = []
for k in os.listdir(path):
    img = os.listdir(path + k)
    temp = []
    for i in img:
        temp.append(path + k + '/' + i)
    images.extend(img)
    img_path.extend(temp)
len(images), len(img_path)


# In[4]:


images_list, age_list, gender_list, race_list, temp, image_path = [], [], [], [], [], []
for i, j in zip(images, img_path):
    try:
        age, gender, race = int(i.split('_')[0]), int(i.split('_')[1]), int(i.split('_')[2])
        image_path.append(j)
        images_list.append(i)
        age_list.append(age)
        gender_list.append(gender)
        race_list.append(race)
    except:
        print(i)
        temp.append(i)
        print('error')
len(images_list), len(age_list), len(gender_list), len(race_list), len(image_path)


# In[5]:


data = pd.DataFrame()
data['age'], data['gender'], data['race'], data['img'], data['images_list'] = age_list, gender_list, race_list, image_path, images_list
# data = data[(data['age']>30) & (data['age']<32)]
data.shape


# In[6]:


X_train = []
gender_list, age_list, race_list = [], [], []
for i in range(len(data)):
    image = cv2.imread(data.iloc[i][3])
    gender, age, race = data.iloc[i][1], data.iloc[i][0], data.iloc[i][2]
    face, confidence = cv.detect_face(image)
    try: 
        for idx, f in enumerate([face[np.argmax(confidence)]]):
            startX, startY, endX, endY = f[0], f[1], f[2], f[3]
            cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
            face_crop = np.copy(image[startY:endY,startX:endX])
            face_crop = cv2.resize(face_crop, input_shape)
            face_crop = face_crop.astype("float")
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            X_train.append(face_crop)
            gender_list.append(gender)
            age_list.append(age)
            race_list.append(race)
    except:
        print('error', i)


# In[7]:


len(X_train), len(gender_list), len(age_list), len(race_list)


# In[8]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
imgplot = plt.imshow(X_train[0])
plt.show()


# In[10]:


X_train = np.array(X_train)
y = np.array(gender_list)
yage = np.array(age_list)
yrace = np.array(race_list)
X_train.shape, y.shape, yage.shape, yrace.shape


# In[1]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test, yage_train, yage_test, yrace_train, yrace_test  = train_test_split(
    X_train, y, yage , yrace,  test_size=0.20, random_state=42)
print(x_train.shape, y_train.shape, yage_train.shape, yrace_train.shape)
print(x_test.shape, y_test.shape, yage_test.shape, yrace_test.shape)


# In[ ]:


np.save(path_cb + 'data/train/x_train_1_all_cropped_299', x_train)
np.save(path_cb + 'data/train/y_train_1_all_cropped_299', y_train)
np.save(path_cb + 'data/score/x_test_1_all_cropped_299', x_test)
np.save(path_cb + 'data/score/y_test_1_all_cropped_299', y_test)


# In[12]:


np.save(path_cb + 'data/train/yage_train_1_all', yage_train)
np.save(path_cb + 'data/train/yrace_train_1_all', yrace_train)
np.save(path_cb + 'data/train/yage_test_1_all', yage_test)
np.save(path_cb + 'data/train/yrace_test_1_all', yrace_test)


# ### Temp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
imgplot = plt.imshow(x_train[0])
plt.show()

