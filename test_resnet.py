#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:30:51 2018

@author: fenqiang
"""

import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os


from tensorboardX import SummaryWriter
writer = SummaryWriter('log/f')

from model import ResNet

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, root2 = None, root3 = None, root4 = None):

        self.files = sorted(glob.glob(os.path.join(root1, '*')))    
        for i in range(len(self.files)):
            if os.path.exists('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold2/' + self.files[i].split('/')[-1].split('_')[2] + '.Y1'):
                self.files.append('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold2/' + self.files[i].split('/')[-1].split('_')[2] + '.Y1')
            elif os.path.exists('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold1/' + self.files[i].split('/')[-1].split('_')[2] + '.Y1'):
                self.files.append('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold1/' + self.files[i].split('/')[-1].split('_')[2] + '.Y1')
            else:
                print(self.files(i))

    def __getitem__(self, index):
        file = self.files[index]
        if file.split('.')[-1] == 'txt':
            data = np.loadtxt(file)
            data = data[:, np.newaxis]
            label = 0.0
        elif file.split('.')[-1] == 'Y1':
            data = sio.loadmat(file)
            data = data['data'][:,[2]]
            label = 1.0
        
        return data.astype(np.float32), float(label)

    def __len__(self):
        return len(self.files)


cuda = torch.device('cuda:0') 
learning_rate = 0.0001
batch_size = 1
fold1 = '/home/fenqiang/Spherical_U-Net/pred/train'
fold2 = '/home/fenqiang/Spherical_U-Net/pred/val'

train_dataset = BrainSphere(fold1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


model = ResNet(1, 1)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
L = nn.BCELoss()


def val_during_training(dataloader):
    model.eval()

    num = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze(0).cuda(cuda)
        with torch.no_grad():
            prediction = model(data).squeeze()
        if (prediction > 0.5 and target == 1) or (prediction <= 0.5 and target == 0):
            num = num+1

    return num/len(dataloader)


def get_learning_rate(epoch):
    limits = [3, 6, 8]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate

for epoch in range(300):

    train_acc  = val_during_training(train_dataloader)
    val_acc  = val_during_training(val_dataloader)
    print("Train acc, val acc : {:.4} {:.4}".format(train_acc, val_acc))
    
    lr = get_learning_rate(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):

        model.train()
        data, target = data.squeeze(0).cuda(cuda), target.squeeze().cuda(cuda).float()
    
        prediction = model(data).squeeze()
    
        loss = L(prediction, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
        
     
    torch.save(model.state_dict(), "trained_models/resnet_for_real_LRpred.pkl")
