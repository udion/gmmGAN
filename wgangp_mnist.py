#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.autograd as agd
import torch.nn as tchnn
import torch.nn.functional as F
import torch.optim as optim
import random

import os, sys
import random
import visdom

vis = visdom.Visdom(port=7777)

#some params
DIM = 2048 #bigger than 728, mimicing the model of 2 d gaussian
FIXED_GEN = False
LAMBDA = 10
DISCRI_ITR = 5
BATCHSZ = 256
batchSz = BATCHSZ
TOT_GEN_ITR = 1000#100000

#models of generator and discriminator gen has 28X28 i/o Discri has 28X28 input nodes and 
class Generator(tchnn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.L1 = tchnn.Linear(784, DIM)
        self.L2 = tchnn.Linear(DIM, DIM)
        self.L3 = tchnn.Linear(DIM, DIM)
        self.Ou = tchnn.Linear(DIM,784)
    def forward(self, noise, real_data):
        if FIXED_GEN:
            return noise + real_data
        else:
            x = F.relu(self.L1(noise))
            x = F.relu(self.L2(x))
            x = F.relu(self.L3(x))
            x = self.Ou(x)
            return x.view(-1,784)
    def name(self):
        return 'GENERATOR'

class Discriminator(tchnn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1 = tchnn.Linear(784, DIM)
        self.L2 = tchnn.Linear(DIM, DIM)
        self.L3 = tchnn.Linear(DIM, DIM)
        self.Ou = tchnn.Linear(DIM,1)
    def forward(self, x):
            x = F.relu(self.L1(x))
            x = F.relu(self.L2(x))
            x = F.relu(self.L3(x))
            x = self.Ou(x)
            return x.view(-1)
    def name(self):
        return 'DISCRIMINATOR'



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#gradient penalty term in objective
def calc_gp(D, real_data, fake_data):
    a = torch.rand(BATCHSZ, 1)
    a = a.expand(real_data.size())
    a = a.cuda()
    
    interpolated = a*real_data + (1-a)*fake_data
    interpolated = interpolated.cuda()
    interpolated = agd.Variable(interpolated, requires_grad=True)
    
    D_interp = D(interpolated)
    
    gradients = agd.grad(outputs=D_interp, inputs=interpolated, grad_outputs=torch.ones(D_interp.size()).cuda()
                        ,create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1)**2).mean()*LAMBDA
    return gp

#data loadin and stuff
MNISTX_train = np.load('MNISTX_train.npy')
def MNIST_gen(X, BATCHSZ):
    X = X.reshape(-1,784) #serialize images
    while(True):
        databatch = random.sample(list(X), BATCHSZ)
        databatch = np.array(databatch)
        yield databatch
MNISTd = MNIST_gen(MNISTX_train, BATCHSZ)

#instantiating the networks and optimizers and stuff
#notaion similar to paper
G = Generator().cuda()
D = Discriminator().cuda()
G.apply(weights_init)
D.apply(weights_init)
print(G)
print(D)

one = torch.FloatTensor([1])
onebar = one * -1
one = one.cuda()
onebar = onebar.cuda()
optD = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
optG = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))


#plotter for mnist
def plotter(batch_data):
    #batch_data = batch_data.numpy()
    n = batch_data.shape[0]
    for i in range(n):
        plt.subplot(3,3,i+1)
        plt.imshow(batch_data[i].reshape(-1,28), cmap='gray', interpolation='none')
        plt.axis('off')
    plt.show()




#trainer -- dont touch unless you want to train
for itr in range(10000):
    ## D training and hence layers paprams should get updated.
    for p in D.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in range(DISCRI_ITR):
        #print(iter_d)
        _data = next(MNISTd)
        real_data = torch.Tensor(_data)
        real_data = real_data.cuda()
        real_data_v = agd.Variable(real_data)
        #print(real_data_v)

        D.zero_grad()

        # train with real
        D_real = D(real_data_v)
        D_real = D_real.mean()
        ##D_real.backward(onebar)

        # train with fake
        noise = torch.randn(BATCHSZ, 784)
        noise = noise.cuda()
        noisev = agd.Variable(noise, volatile=True)  # totally freeze netG
        
        #noisev = agd.Variable(noise)
        gop = G(noisev, real_data_v)
        #print('here3')
        fake = agd.Variable(gop.data)
        inputv = fake
        D_fake = D(inputv)
        D_fake = D_fake.mean()
        ##D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gp(D, real_data_v.data, fake.data)
        ##gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        D_cost.backward()
        Wasserstein_D = D_real - D_fake
        optD.step()
        #print('discri iter done ', iter_d)
    fake2plot = ''
    if not FIXED_GEN:
        #now the discriminator should not be updating it's weights
        for p in D.parameters():
            p.requires_grad = False  # to avoid computation
        G.zero_grad()

        _data = next(MNISTd)
        real_data = torch.Tensor(_data)
        real_data = real_data.cuda()
        real_data_v = agd.Variable(real_data)

        noise = torch.randn(BATCHSZ, 784)
        noise = noise.cuda()
        noisev = agd.Variable(noise)
        fake = G(noisev, real_data_v)
        fake2plot = fake.cpu().data.numpy()
        g_ = D(fake)
        g_ = g_.mean()
        g_.backward(onebar)
        G_cost = -g_
        optG.step()
    
    if not FIXED_GEN:
        if itr % 10 == 0:
            print('training till itr: {} done'.format(itr))
            fake2plot = fake2plot.reshape(-1,1,28,28)
            f1 = list(fake2plot)
            f2 = random.sample(f1, 64)
            f3 = np.array(f2)
            #plotter(fake2plot[0:9,:])
            vis.images(f3, 
            	opts=dict(title='after itr:{}'.format(itr), caption='showing 64 randomly chosen generator output after itr {}:'.format(itr)))










