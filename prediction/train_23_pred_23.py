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

from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')


from models import *
from spherical_gan import Generator

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
        feats_Y0 = raw_Y0['data'][:,[1,2]]
        feats_Y0 = (feats_Y0 - np.tile(np.min(feats_Y0,0), (len(feats_Y0),1)))/(np.tile(np.max(feats_Y0,0), (len(feats_Y0),1)) - np.tile(np.min(feats_Y0,0), (len(feats_Y0),1)))
        feats_Y0 = feats_Y0 * 2.0 - 1.0
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        raw_Y1 = raw_Y1['data'][:,[1,2]]   # 0: curv, 1: sulc, 2: thickness
        feats_Y1 = (raw_Y1 - np.tile(np.min(raw_Y1,0), (len(raw_Y1),1)))/(np.tile(np.max(raw_Y1,0), (len(raw_Y1),1)) - np.tile(np.min(raw_Y1,0), (len(raw_Y1),1)))
        feats_Y1 = feats_Y1 * 2.0 - 1.0
        
        return feats_Y0.astype(np.float32), feats_Y1.astype(np.float32), raw_Y1.astype(np.float32)

    def __len__(self):
        return len(self.files)

cuda0 = torch.device('cuda:1') 
learning_rate = 0.00005
momentum = 0.99
weight_decay = 0.0001
batch_size = 1
fold1 = '/media/fenqiang/DATA/unc/Data/from_dingna371/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/from_dingna371/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/from_dingna371/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/from_dingna371/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/from_dingna371/fold5'
train_dataset = BrainSphere(fold1, fold2, fold3, fold4)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold5)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

conv_type = "DiNe"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = Generator(2, conv_type, pooling_type)


print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda0)
mseloss = nn.MSELoss()
l1Loss = nn.L1Loss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.8, 0.999))
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel')


def train_step(data, target):
    model.train()
    data, target = data.cuda(cuda0), target.cuda(cuda0)

    prediction = model(data)
    
    loss = mseloss(prediction, target) + 5 * l1Loss(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def get_learning_rate(epoch):
    limits = [2, 4, 7]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

def val_during_training():
    model.eval()

    all_error_sulc = torch.tensor([])
    all_error_thick = torch.tensor([])
    for batch_idx, (data, _, raw_target) in enumerate(val_dataloader):
        data = data.squeeze().cuda(cuda0)
        raw_target = raw_target.squeeze()
        with torch.no_grad():
            prediction = model(data)
        for i in range(2):
            pred = prediction[:,i].cpu() 
            feat_Y1 = raw_target[:,i]
            pred = (pred + 1.0) / 2.0 * (feat_Y1.max() - feat_Y1.min()) + feat_Y1.min()
            if i == 0:
                all_error_sulc = torch.cat((all_error_sulc, torch.abs(pred - feat_Y1)), 0)
            else:
                all_error_thick = torch.cat((all_error_thick, torch.abs(pred - feat_Y1)), 0)

    return torch.mean(all_error_sulc), torch.std(all_error_sulc),torch.mean(all_error_thick), torch.std(all_error_thick) 


def val_train_during_training():
    model.eval()

    all_error_sulc = torch.tensor([])
    all_error_thick = torch.tensor([])
    for batch_idx, (data, _, raw_target) in enumerate(train_dataloader):
        data = data.squeeze().cuda(cuda0)
        raw_target = raw_target.squeeze()
        with torch.no_grad():
            prediction = model(data)
        for i in range(2):
            pred = prediction[:,i].cpu() 
            feat_Y1 = raw_target[:,i]
            pred = (pred + 1.0) / 2.0 * (feat_Y1.max() - feat_Y1.min()) + feat_Y1.min()
            if i == 0:
                all_error_sulc = torch.cat((all_error_sulc, torch.abs(pred - feat_Y1)), 0)
            else:
                all_error_thick = torch.cat((all_error_thick, torch.abs(pred - feat_Y1)), 0)

    return torch.mean(all_error_sulc), torch.std(all_error_sulc),torch.mean(all_error_thick), torch.std(all_error_thick) 


for epoch in range(300):
    
    train_sulc_mean, train_sulc_std, train_thic_mean, train_thic_std = val_train_during_training()
    print("train_sulc mean+-std: {:.4} {:.4}".format(train_sulc_mean, train_sulc_std))
    print("train_thic mean+-std: {:.4} {:.4}".format(train_thic_mean, train_thic_mean))
    
    val_sulc_mean, val_sulc_std, val_thic_mean, val_thic_std = val_during_training()
    print("val_sulc mean+-std: {:.4} {:.4}".format(val_sulc_mean, val_sulc_std))
    print("val_thic mean+-std: {:.4} {:.4}".format(val_thic_mean, val_thic_std))
    
    writer.add_scalars('sulc/t', {'train': train_sulc_mean, 'val': val_sulc_mean}, epoch)
    writer.add_scalars('thick/w', {'train': train_thic_mean, 'val': val_thic_mean}, epoch)


    lr = get_learning_rate(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target, raw_target = dataiter.next()
    
    for batch_idx, (data, target, _) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
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
    torch.save(model.state_dict(), os.path.join("use23_pred23.pkl"))
#  




#%%

#all_error = torch.tensor([])
#for batch_idx, (data, _, raw_target) in enumerate(val_dataloader):
#    prediction = data.squeeze().cpu() 
#    raw_target = raw_target.squeeze()
#    prediction = (prediction + 1.0) / 2.0 * (raw_target.max() - raw_target.min()) + raw_target.min()
#    all_error = torch.cat((all_error, torch.abs(prediction - raw_target)), 0)
#    
#mean = torch.mean(all_error)
#std = torch.std(all_error)
    
    
