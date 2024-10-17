#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network
# - MNIST data
# - 3 convolutional layers
# - 2 fully connected layers

# ## 1. Settings
# ### 1) Import required libraries

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable, Function
from visdom import Visdom
viz = Visdom()


# In[2]:


class Swish(Function):
    @staticmethod
    def forward(ctx, i):
        result = i*i.sigmoid()
        ctx.save_for_backward(result,i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result,i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result+sigmoid_x*(1-result))
    
swish= Swish.apply

class Swish_module(nn.Module):
    def forward(self,x):
        return swish(x)
    
swish_layer = Swish_module()


# ### 2) Set hyperparameters

# In[3]:


batch_size = 256
learning_rate = 0.0002
num_epoch = 2


# ## 2. Data
# 
# ### 1) Download Data

# In[4]:


mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


# ### 2) Check Dataset

# In[5]:


print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
mnist_test.__getitem__(0)[0].size(), mnist_test.__len__()


# ### 3) Set DataLoader

# In[6]:


train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)


# ## 3. Model & Optimizer
# 
# ### 1) CNN Model

# In[7]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5),
            swish_layer,
            #nn.ReLU(),
            nn.Conv2d(16,32,5),
            swish_layer,
            #nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            swish_layer,
            #nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            swish_layer,
            #nn.ReLU(),
            nn.Linear(100,10)
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)

        return out

model = CNN().cuda()


# ### 2) Loss func & Optimizer

# In[8]:


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ## 4. Train 

# In[9]:


for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = Variable(image).cuda()
        y_= Variable(label).cuda()
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            print(loss)          


# In[10]:


#param_list = list(model.parameters())
#print(param_list)


# ## 5. Test

# In[11]:


# swish: 93.3193 / 91.5966 /  92.5481 / 96.1538 / 93.4996
# relu: 90.8353 / 88.8722 / 91.3962 / 93.6198 / 93.2492

correct = 0
total = 0

for image,label in test_loader:
    x = Variable(image,volatile=True).cuda()
    y_= Variable(label).cuda()

    output = model.forward(x)
    _,output_index = torch.max(output,1)
        
    total += label.size(0)
    correct += (output_index == y_).sum().float()
    
print("Accuracy of Test Data: {}".format(100*correct/total))

