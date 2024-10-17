#!/usr/bin/env python
# coding: utf-8

# # InfoGAN 
# 
# - Discrete category + continuous category
# 
# <img src="./infogan.png" width="400">

# ## 1. Import required libraries

# In[1]:


import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# ## 2. Hyperparameter & Data setting

# In[2]:


# Set Hyperparameters

epoch = 50
batch_size = 128
learning_rate = 0.0002
num_gpus = 1
z_size= 62
discrete_latent_size = 10
contin_latent_size = 2
ratio = 1

# Download Data & Set Data Loader(input pipeline)

mnist_train = dset.MNIST("./", train=True, 
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]),
                        target_transform=None,
                        download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)


# ## 3. Label into one-hot vector

# In[3]:


def int_to_onehot(z_label):
    one_hot_array = np.zeros(shape=[len(z_label), discrete_latent_size])
    one_hot_array[np.arange(len(z_label)), z_label] = 1
    return one_hot_array


# ## 4. Generator

# In[4]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Linear(z_size+discrete_latent_size+contin_latent_size,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),               
                    nn.Linear(1024,7*7*256),               
                    nn.BatchNorm1d(7*7*256),
                    nn.ReLU(), 
            )              
        self.layer2 = nn.Sequential(OrderedDict([
                ('conv1', nn.ConvTranspose2d(256,128,3,2,1,1)), # [batch,256,7,7] -> [batch,128,14,14]
                ('bn1', nn.BatchNorm2d(128)),    
                ('relu1', nn.ReLU()),
                ('conv2', nn.ConvTranspose2d(128,64,3,1,1)),    # [batch,128,14,14] -> [batch,64,14,14]
                ('bn2', nn.BatchNorm2d(64)),    
                ('relu2', nn.ReLU()),
                
            ]))
        self.layer3 = nn.Sequential(OrderedDict([
                ('conv3',nn.ConvTranspose2d(64,32,3,1,1)),      # [batch,64,14,14] -> [batch,16,14,14]
                ('bn3',nn.BatchNorm2d(32)),    
                ('relu3',nn.ReLU()),
                ('conv4',nn.ConvTranspose2d(32,1,3,2,1,1)),     # [batch,16,14,14] -> [batch,1,28,28]
                ('tanh',nn.Tanh())
            ]))

    def forward(self,z):
        out = self.layer1(z)
        out = out.view(batch_size//num_gpus,256,7,7)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# ## 5. Discriminator

# In[5]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
                ('conv1',nn.Conv2d(1,32,3,stride=2,padding=1)),        # [batch,1,28,28] -> [batch,32,14,14]
                ('relu1',nn.LeakyReLU(0.1)),
                ('conv2',nn.Conv2d(32,64,3,stride=2,padding=1)),      # [batch,32,14,14] -> [batch,64,7,7]
                ('bn2',nn.BatchNorm2d(64)),    
                ('relu2',nn.LeakyReLU(0.1)),
            ]))
        
        self.layer2 = nn.Sequential(
                nn.Linear(64*7*7,256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Linear(256,1+discrete_latent_size+contin_latent_size) # GAN + Category + Continuous
            )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        out = self.layer1(x)
        out = out.view(batch_size//num_gpus, -1)
        
        out = self.layer2(out)  
                
        output = self.sigmoid(out[:,0:1])
        onehot = self.sigmoid(out[:,1:11])
        contin = out[:,11:]
        
        return output,onehot,contin


# ## 6. Instance & Label on GPU

# In[6]:


# put class instance on multi gpu

generator = nn.DataParallel(Generator(),device_ids=[0])
discriminator = nn.DataParallel(Discriminator(),device_ids=[0])

# put labels on multi gpu

ones_label = Variable(torch.ones(batch_size,1)).cuda()
zeros_label = Variable(torch.zeros(batch_size,1)).cuda()


# ## 7. Loss function & Optimizer

# In[7]:


gan_loss_func = nn.BCELoss()
cat_loss_func = nn.CrossEntropyLoss()
contin_loss_func = nn.MSELoss()

gen_optim = torch.optim.Adam(generator.parameters(), lr= 5*learning_rate,betas=(0.5,0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))


# ## 8. Model restore

# In[8]:


try:
    generator, discriminator = torch.load('./model/infogan_catcon.pkl')
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass


# In[9]:


def image_check(gen_fake):
    img = gen_fake.data.numpy()
    for i in range(10):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()

def contin_check(i):
    for j in range(10):
        z_random = np.random.rand(batch_size,z_size)
        z_onehot = np.random.randint(0, 10, size=batch_size)
        z_contin = np.random.uniform(-1,1,size=[batch_size,2])

        # change first 10 labels from random to 0~9          
        for l in range(40):
            z_onehot[l]=j

            if l <= 20: 
                z_contin[l,0]= (l-10)/5
                z_contin[l,1]= 0 
                
            else:      
                z_contin[l,0]= 0
                z_contin[l,1]= (l-30)/5

        #print(z_contin)

        # preprocess z
        z_label_onehot = int_to_onehot(z_onehot)
        z_concat = np.concatenate([z_random, z_label_onehot,z_contin], axis=1)
        z = Variable(torch.from_numpy(z_concat).type_as(torch.FloatTensor())).cuda()

        gen_fake = generator.forward(z)

        v_utils.save_image(gen_fake.data[0:40],"./result_contin/gen_{}_{}.png".format(i,j),nrow=10)


# ## 9. Train Model

# In[10]:


for i in range(epoch):
    for j,(image,_) in enumerate(train_loader):
        
        # put image & label on gpu
        image = Variable(image).cuda()
    
        #####################
        ##  discriminator  ##
        ##################### 
        
        z_random = np.random.normal(0,0.1,size=[batch_size,z_size])
        z_onehot = np.random.randint(0, 10, size=batch_size)
        z_contin = np.random.uniform(-1,1,size=[batch_size,2])

        # change first 10 labels from random to 0~9          
        
        for l in range(10):
            z_onehot[l]=l

        # preprocess z
        
        z_label_onehot = int_to_onehot(z_onehot)
        z_concat = np.concatenate([z_random, z_label_onehot,z_contin], axis=1)

        z = Variable(torch.from_numpy(z_concat).type_as(torch.FloatTensor())).cuda()
        z_label_category = Variable(torch.from_numpy(z_onehot).type_as(torch.LongTensor())).cuda()
        z_label_contin = Variable(torch.from_numpy(z_contin).type_as(torch.FloatTensor())).cuda()
       
        # dis_loss = gan_loss(fake & real) + categorical loss
        
        gen_fake = generator.forward(z)
        dis_fake, onehot_fake, contin_fake = discriminator.forward(gen_fake)
        
        dis_optim.zero_grad()
        dis_real, label_real, contin_real = discriminator.forward(image)
        dis_loss = torch.sum(gan_loss_func(dis_fake,zeros_label))\
                 + torch.sum(gan_loss_func(dis_real,ones_label))\
                 + ratio *(torch.sum(cat_loss_func(onehot_fake,z_label_category))\
                 + torch.sum(contin_loss_func(contin_fake,z_label_contin)))
                
        dis_loss.backward()
        dis_optim.step()
        
        #################
        ##  generator  ##
        #################
            
        z_random = np.random.normal(0,0.1,size=[batch_size,z_size])
        z_onehot = np.random.randint(0, 10, size=batch_size)
        z_contin = np.random.uniform(-1,1,size=[batch_size,2])

        # change first 10 labels from random to 0~9   
        
        for l in range(10):
            z_onehot[l]=l

        # preprocess z
        
        z_label_onehot = int_to_onehot(z_onehot)
        z_concat = np.concatenate([z_random, z_label_onehot,z_contin], axis=1)

        z = Variable(torch.from_numpy(z_concat).type_as(torch.FloatTensor())).cuda()
        z_label_category = Variable(torch.from_numpy(z_onehot).type_as(torch.LongTensor())).cuda()
        z_label_contin = Variable(torch.from_numpy(z_contin).type_as(torch.FloatTensor())).cuda()


        # gen_loss = gan loss(fake) + categorical loss
        
        gen_optim.zero_grad()
        gen_fake = generator.forward(z)
        dis_fake, onehot_fake, contin_fake = discriminator.forward(gen_fake)
        
        
        gen_loss = torch.sum(gan_loss_func(dis_fake,ones_label)) \
                 + ratio *(torch.sum(cat_loss_func(onehot_fake,z_label_category))\
                 + torch.sum(contin_loss_func(contin_fake,z_label_contin)))

        gen_loss.backward()
        gen_optim.step()

        # model save
        
        if j % 10 == 0:
            torch.save([generator,discriminator],'./model/infogan_catcon.pkl')

            # print loss and image save
            print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
            v_utils.save_image(gen_fake.data[0:20],"./result_catcon/gen_{}_{}.png".format(i,j), nrow=5)
        
    image_check(gen_fake.cpu())
    contin_check(i)

