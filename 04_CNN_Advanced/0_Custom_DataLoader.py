#!/usr/bin/env python
# coding: utf-8

# # Custom Data Loader

# ## 1. Settings
# ### 1) Import required libraries

# In[1]:


import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms


# ### 2) Hyperparameter

# In[5]:


batch_size = 2


# ## 3) ImageFolder & DataLoader

# In[6]:


img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True,drop_last=True)


# ## 4) Test

# In[7]:


for image,label in img_batch:
    print(image.size(),label)

