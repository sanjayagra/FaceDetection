
# coding: utf-8

# In[2]:


import pandas as pd, os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
path = '/home/ec2-user/Sanjay/projects/6_FaceDetection/Data/'
path_cb = '/home/ec2-user/Sanjay/projects/6_FaceDetection/1_gender/'
input_shape = (224, 224)


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
# data = data[(data['age']>30) & (data['age']<35)]
data.shape


# In[6]:


X_train = []
for i in range(len(data)):
    print(i)
    image = load_img(data.iloc[i][3], target_size=input_shape)
    image = img_to_array(image).astype('float32')
    X_train.append(image)


# In[7]:


X_train = np.array(X_train)
X_train = X_train/255.0
y = data['gender'].values
yage = data['age'].values
yrace = data['race'].values
X_train.shape, y.shape, yage.shape, yrace.shape


# In[8]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test  = train_test_split(X_train, y, test_size=0.20, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[9]:


np.save(path_cb + 'data/train/x_train_1_all', x_train)
np.save(path_cb + 'data/train/y_train_1_all', y_train)
np.save(path_cb + 'data/score/x_test_1_all', x_test)
np.save(path_cb + 'data/score/y_test_1_all', y_test)

