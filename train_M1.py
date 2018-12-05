#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:29:28 2018

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
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

from model import Unet

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, root2 = None, root3 = None, root4 = None):

        self.files = sorted(glob.glob(os.path.join(root1, '*.Y0')))    
        if root2 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root2, '*.Y0')))
        if root3 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root3, '*.Y0')))
        if root4 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root4, '*.Y0')))

    def __getitem__(self, index):
        file = self.files[index]
        raw_Y0 = sio.loadmat(file)
        raw_Y0 = raw_Y0['data'][:,[0, 1, 2]]
#        feats_Y0 = (raw_Y0 - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))/(np.tile(np.max(raw_Y0,0), (len(raw_Y0),1)) - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))
#        feats_Y0 = feats_Y0 * 2.0 - 1.0
#        feats_Y0 = (raw_Y0 - np.tile([0.0081, 1.8586], (len(raw_Y0),1)))/np.tile([0.5169, 0.4395], (len(raw_Y0),1))
        
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        raw_Y1 = raw_Y1['data'][:,2]   # 0: curv, 1: sulc, 2: thickness, 3: x, 4: y, 5: z
#        feats_Y1 = (raw_Y1 - np.tile(np.min(raw_Y1), len(raw_Y1)))/(np.tile(np.max(raw_Y1), len(raw_Y1)) - np.tile(np.min(raw_Y1), len(raw_Y1)))
#        feats_Y1 = feats_Y1 * 2.0 - 1.0
        
        return raw_Y0.astype(np.float32), raw_Y1.astype(np.float32)

    def __len__(self):
        return len(self.files)

cuda = torch.device('cuda:1') 
learning_rate = 0.0001
momentum = 0.99
#weight_momentum = 0.95
weight_decay = 0.0001
batch_size = 1

fold1 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold5'

train_dataset = BrainSphere(fold2, fold1, fold5, fold3)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

conv_type = "DiNe"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = Unet(3, 1)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel')

#vertex_weight = torch.ones(40962).cuda(cuda)
#plt.axis([0, 40962, 0, 2.5])
#plt.ion()
def train_step(data, target):
    
    model.train()
    data, target = data.squeeze(0).cuda(cuda), target.squeeze(0).cuda(cuda)

    prediction = model(data).squeeze()
#    prediction = (prediction + 1.0) / 2.0 * (raw_target.max() - raw_target.min()) + raw_target.min()
    
#    L0 = torch.mean(torch.abs(prediction - target)/target)
    L1 = torch.mean(torch.abs(prediction - target))
#    L2 = torch.mean((prediction - target) ** 2)
#    loss = (torch.dot((prediction - target) ** 2, vertex_weight) + 4 * torch.dot(torch.abs(prediction - target), vertex_weight))/40962.0
    loss =  L1

#    vertex_weight_update(prediction.detach(), target.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

#def vertex_weight_update(prediction, target):
#    
#    global vertex_weight
#    plt.clf()
#    plt.plot(np.arange(40962), vertex_weight.cpu().numpy())
#    plt.pause(0.01)
#    
#    w1 = torch.abs(prediction - target) / torch.mean(torch.abs(prediction - target))  # mean = 1, min >= 0, max = infinite
#    w2 = (prediction - target) ** 2 / torch.mean((prediction - target) ** 2)          # mean = 1, min >= 0, max = infinite
#    weight_update = 0.8 * w1 + 0.2 * w2                                               # mean = 1, min >= 0
#    vertex_weight = weight_momentum * vertex_weight + (1 - weight_momentum) * weight_update


def get_learning_rate(epoch):
    limits = [3, 6, 8]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

def val_during_training(dataloader):
    model.eval()

    mae = torch.tensor([])
    mre = torch.tensor([])
    for batch_idx, (data, raw_target) in enumerate(dataloader):
        data = data.squeeze(0).cuda(cuda)
        with torch.no_grad():
            prediction = model(data)
        prediction = prediction.squeeze().cpu() 
        raw_target = raw_target.squeeze(0).cpu()
#        prediction = (prediction + 1.0) / 2.0 * (raw_target.max() - raw_target.min()) + raw_target.min()
        mae = torch.cat((mae, torch.abs(prediction - raw_target)), 0)
        mre = torch.cat((mre, torch.abs(prediction - raw_target)/raw_target), 0)
        
    m_mae, s_mae = torch.mean(mae), torch.std(mae)
    m_mre, s_mre= torch.mean(mre), torch.std(mre)

    return m_mae, s_mae, m_mre, s_mre


for epoch in range(300):
    
    m_mae, s_mae, m_mre, s_mre  = val_during_training(train_dataloader)
    print("Train: mean mae, std mae, mean mre, std mre : {:.4} {:.4} {:.4} {:.4}".format(m_mae, s_mae, m_mre, s_mre))
    writer.add_scalars('data/mean', {'train': m_mae}, epoch)
    m_mae, s_mae, m_mre, s_mre = val_during_training(val_dataloader)
    print("Val: mean mae, std mae, mean mre, std mre : {:.4} {:.4} {:.4} {:.4}".format(m_mae, s_mae, m_mre, s_mre))    
    writer.add_scalars('data/mean', {'val': m_mae}, epoch)

    lr = get_learning_rate(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, raw_target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):

        loss = train_step(data, target)

        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)


    
#    scheduler.step(scheduler_loss)
#
#    train_dice[epoch % 5] = a
#    print("last five train Dice: ",train_dice)
#    if np.std(np.array(train_dice)) <= 0.00001:
#        torch.save(model.state_dict(), os.path.join("state.pkl"))
#        break
#
 
#    torch.save(model.state_dict(), os.path.join("missingthickness.pkl"))
