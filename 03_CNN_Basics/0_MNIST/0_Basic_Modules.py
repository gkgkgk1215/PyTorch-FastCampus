#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network
# - MNIST data
# - Convolution Layer
# - Pooling Layer

# ## 1. Settings
# ### 1) Import required libraries

# In[1]:


import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.datasets.mnist
import torchvision.transforms as transforms
from IPython.conftest import get_ipython
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# ## 2. Data
# 
# ### 1) Download Data

# In[2]:


mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)


# ### 2) Item

# In[3]:


print(mnist_train)

# dataset.__getitem__(idx)
image,label = mnist_train.__getitem__(0)
print(image.size(),label)
# dataset[idx]
image,label = mnist_train[0]
print(image.size(),label)


# ### 3) Length

# In[4]:


# dataset.__len__()
print(mnist_train.__len__())

# len(dataset)
len(mnist_train)


# ### 4) Show Image

# In[5]:


for i in range(3):
    img= mnist_train[i][0].numpy()
    plt.imshow(img[0],cmap='gray')
    plt.show()


# ## 3. Convolution Layer
# 
# - torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# - Channels
# - Kernel size
# - Stride
# - Padding
# - [Batch,Channel,Height,Width]
# 
# 
# ### 1) Channels

# In[6]:


image,label = mnist_train[0]
image = image.view(-1,image.size()[0],image.size()[1],image.size()[2])

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3)
output = conv_layer(Variable(image))
print(output.size())

for i in range(3):
    plt.imshow(output[0,i,:,:].data.numpy(),cmap='gray')
    plt.show()


# ### 2) Kernel Size

# In[7]:


conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())


# ### 3) Stride

# In[8]:


conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,stride=1)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=2)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5,stride=3)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())


# ### 4) Padding

# In[9]:


conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,padding=1)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,padding=1)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

conv_layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5,padding=1)
output = conv_layer(Variable(image))
plt.imshow(output[0,0,:,:].data.numpy(),cmap='gray')
plt.show()
print(output.size())

