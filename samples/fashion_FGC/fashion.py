#!/usr/bin/env python
# coding: utf-8

# In[70]:


import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf


# In[71]:


DATA_DIR = Path(r'/home/aditya/Downloads/Mask_RCNN/fashion_dataset')
ROOT_DIR = Path(r'/home/aditya/Downloads/Mask_RCNN')


# In[72]:


sys.path.append(r'/home/aditya/Downloads/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils_for_FGC
import mrcnn.model_for_FGC as modellib
from mrcnn import visualize
from mrcnn.model_for_FGC import log


# In[ ]:





# In[73]:


COCO_WEIGHTS_PATH = r'/home/aditya/Downloads/Mask_RCNN/mask_rcnn_coco.h5'
NUM_CATS = 46
IMAGE_SIZE = 1024


# In[74]:


class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 # a memory error occurs when IMAGES_PER_GPU is too high
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

    ## My changes CA
    BACKBONE = 'resnet101'
    
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024    
    IMAGE_RESIZE_MODE = 'square'

    MINI_MASK_SHAPE = (112, 112)  # (height, width) of the mini-mask

    NUM_ATTR = 294

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
        "mrcnn_attr_loss":1.
    }


    
config = FashionConfig()
config.display()


# In[75]:


with open(DATA_DIR/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

class_names = [x['name'] for x in label_descriptions['categories']]
attr_names = [x['name'] for x in label_descriptions['attributes']]


# In[76]:


print(len(class_names),len(attr_names))


# In[77]:


segment_df = pd.read_csv(DATA_DIR/"train_small.csv")
segment_df['AttributesIds'] = segment_df['AttributesIds'].apply(lambda x:tuple([int(i) for i in x.split(',')]))


# In[78]:


def pad_tuple_attrs(x):
    if x!=x:
        x = []
    else:
        x = list(x)
    for i in range(10):
        if(i>=len(x)):
            x.append(-1)
        if x[i]>=281 and x[i]<284:
            x[i] = x[i]-46
        elif x[i]>284:
            x[i] = x[i]-47
    
    x = tuple(x)
    return x


# In[79]:


segment_df['AttributesIds'] = segment_df['AttributesIds'].apply(pad_tuple_attrs)


# In[80]:


segment_df['AttributesIds'].head()


# In[81]:


image_df = segment_df.groupby('ImageId')['EncodedPixels', 'ClassId', 'AttributesIds'].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
image_df = image_df.join(size_df, on='ImageId')

print("Total images: ", len(image_df))
image_df.head()


# In[82]:


c = segment_df.iloc[0]['EncodedPixels']
# c= c.split(',')
print(type(c))


# In[83]:


def resize_image(image_path):
    image_path = image_path + ".jpg"
    # print("image_path", image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img


# In[84]:


class FashionDataset(utils_for_FGC.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(class_names):
            self.add_class("fashion", i+1, name)
        
        for i, name in enumerate(attr_names):
            self.add_attribute("fashion", i, name)
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(DATA_DIR/'train'/row.name), 
                           labels=row['ClassId'],
                           attributes=row['AttributesIds'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    

    def image_reference(self, image_id):
        attr_sublist=[]
        attr_list=[]
        info = self.image_info[image_id]
        for x in info['attributes']:
            for j in x:
                if(j>234):
                    j=j-46
                
                attr_sublist.append(attr_names[int(j)])
            attr_list.append(attr_sublist)
            
        return info['path'], [class_names[int(x)] for x in info['labels']],attr_list
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        attributes = []
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            attributes.append(list(info['attributes'][m]))
            
        return mask, np.array(labels), np.array([np.array(attr) for attr in attributes])


# In[85]:


dataset = FashionDataset(image_df)
dataset.prepare()

for i in range(1):
    image_id = random.choice(dataset.image_ids)
    print(dataset.image_reference(image_id))
    
    image = dataset.load_image(image_id)
    mask, class_ids, attr_ids = dataset.load_mask(image_id)
    # print("class_ids", class_ids)
    # print("attr_ids", attr_ids)
    # print(type(attr_ids))
    visualize.display_top_masks(image, mask, class_ids, attr_ids, dataset.class_names, dataset.attr_names, limit=4)


# In[86]:


# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 3

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()


# In[87]:


train_segments = np.concatenate(train_df['ClassId'].values).astype(int)
print("Total train images: ", len(train_df))
print("Total train segments: ", len(train_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, class_names, rotation='vertical')
plt.show()

valid_segments = np.concatenate(valid_df['ClassId'].values).astype(int)
print("Total train images: ", len(valid_df))
print("Total validation segments: ", len(valid_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(valid_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, class_names, rotation='vertical')
plt.show()


train2_segments = np.concatenate(train_df['AttributesIds'].values).astype(int).reshape((-1,))
train2_segments = train2_segments[train2_segments!= -1]
# print("Total train images: ", len(valid_df))
print("Total train segments: ", len(train2_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train2_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, attr_names, rotation='vertical')
plt.show()

train2_segments = np.concatenate(valid_df['AttributesIds'].values).astype(int).reshape((-1,))
train2_segments = train2_segments[train2_segments!= -1]
# print("Total train images: ", len(valid_df))
print("Total train segments: ", len(train2_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train2_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, attr_names, rotation='vertical')
plt.show()



# In[88]:


import warnings 
warnings.filterwarnings("ignore")


# In[89]:


mask


# In[90]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


# In[68]:


augmentation = iaa.Sequential([
    iaa.Fliplr(0.5) # only horizontal flip here
])


# In[44]:


# get_ipython().run_cell_magic('time', '', "model.train(train_dataset, valid_dataset,\n            learning_rate=config.LEARNING_RATE*2, # train heads with higher lr to speedup learning\n            epochs=2,\n            layers='heads',\n            augmentation=None)\n\nhistory = model.keras_model.history.history")


# In[29]:


import tensorflow as tf 
print(tf.__version__)


# In[ ]:





# In[ ]:





# In[ ]:




