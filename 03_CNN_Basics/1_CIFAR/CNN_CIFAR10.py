#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network
# - CIFAR-10 dataset

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


batch_size = 256
learning_rate = 0.0002
num_epoch = 100


# ## 2. Data Generation
# 
# ### 1) Download Data

# In[3]:


mnist_train = dset.CIFAR10("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.CIFAR10("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


# ### 2) Check Dataset

# In[ ]:


print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
mnist_test.__getitem__(0)[0].size(), mnist_test.__len__()


# ### 3) Set DataLoader

# In[ ]:


train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)


# ## 3. Model & Optimizer
# 
# ### 1) CNN Model

# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32 x 16 x 16
            
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128 x 8 x 8
            
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256*4*4,200),
            nn.ReLU(),
            nn.Linear(200,10)
        )       
        
    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)

        return out

model = CNN().cuda()


# ### 2) Loss func & Optimizer

# In[ ]:


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ### 3) Load Model

# In[ ]:


try:
    model = torch.load('./cifar_model.pkl')
    print("model restored")
except:
    print("model not restored")


# ## 4. Train 

# In[ ]:


for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = Variable(image).cuda()
        y_= Variable(label).cuda()
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()
        
        if j % 100 == 0:
            print(loss.data,i,j)
            torch.save(model,'./cifar_model.pkl')


# In[ ]:


param_list = list(model.children())
print(param_list)


# ## 5. Test

# In[ ]:


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


# In[ ]:





# In[ ]:




