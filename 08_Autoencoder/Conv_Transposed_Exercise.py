#!/usr/bin/env python
# coding: utf-8

# # Convolution Transposed Exercise
# 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
# 
# ## 1. Import Required Libraries

# In[1]:


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


# ## 2. Input Data

# In[2]:


img = Variable(torch.ones(1,1,3,3))
print(img)


# ## 3. Set All Weights to One

# In[3]:


transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False)

print(transpose.weight.data)

init.constant(transpose.weight.data,1)


# ## Kernel Size=3, stride=1, padding=0, output_padding=0

# In[4]:


transpose(img)


# ## Kernel Size=3, stride=2, padding=0, output_padding=0

# In[5]:


transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
init.constant(transpose.weight.data,1)
transpose(img)


# ## Kernel Size=3, stride=2, padding=1, output_padding=0

# In[6]:


transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False)
init.constant(transpose.weight.data,1)
transpose(img)


# ## Kernel Size=3, stride=2, padding=0, output_padding=1

# In[7]:


transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, output_padding=1, bias=False)
init.constant(transpose.weight.data,1)
transpose(img)


# ## Kernel Size=3, stride=2, padding=1, output_padding=1

# In[10]:


transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
init.constant(transpose.weight.data,1)
transpose(img)

