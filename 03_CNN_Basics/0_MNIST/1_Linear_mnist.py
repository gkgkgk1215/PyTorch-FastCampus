#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network
# - MNIST data
# - only fully connected layers

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
from torch.autograd import Variable


# ### 2) Set hyperparameters

# In[2]:


batch_size = 16
learning_rate = 0.0002
num_epoch = 10


# ## 3. Data Generation
# 
# ### 1) Download Data

# In[3]:


mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


# ### 2) Check Dataset

# In[4]:


print(mnist_train.__getitem__(0)[1], mnist_train.__len__())
mnist_test.__getitem__(0)[1], mnist_test.__len__()


# ### 3) Set DataLoader

# In[5]:


train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)


# ## 4. Model & Optimizer
# 
# ### 1) CNN Model

# In[6]:


class Linear(nn.Module):
    def __init__(self):
        super(Linear,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784,300),
            nn.ReLU(),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU()
        )       
        
    def forward(self,x):
        out = x.view(batch_size,-1)
        out = self.layer(out)

        return out

model = Linear().cuda()


# ### 2) Loss func & Optimizer

# In[7]:


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ## 5. Train 

# In[8]:


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


# ## 6. Test

# In[9]:


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

