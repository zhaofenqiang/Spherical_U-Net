#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:30:17 2019

@author: fenqiang
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os

from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

#from gCNN import * 
#from comparison_model import *
#from UNet_interpolation import *
#from SegNet import *
from model import Unet

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, root2 = None, root3 = None, root4 = None, root5 = None, root6 = None, root7=None):

        self.files = sorted(glob.glob(os.path.join(root1, '*.mat')))    
        if root2 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root2, '*.mat')))
        if root3 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root3, '*.mat')))
        if root4 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root4, '*.mat')))
        if root5 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root5, '*.mat')))
        if root6 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root6, '*.mat')))
        if root7 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root7, '*.mat')))


    def __getitem__(self, index):
        file = self.files[index]
        data = sio.loadmat(file)
        data = data['data']
        
        feats = data[:,[0,1]]
        feat_max = np.max(feats,0)
        for i in range(np.shape(feats)[1]):
            feats[:,i] = feats[:, i]/feat_max[i]
#        
        label = data[:,5]-1
        return feats.astype(np.float32), label.astype(np.long)

    def __len__(self):
        return len(self.files)


learning_rate = 0.0002
momentum = 0.99
weight_decay = 0.0001
batch_size = 1
cuda = torch.device('cuda:0')

fold1 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigCurv/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigCurv/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigCurv/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigCurv/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigAugmentation/fold1'
fold6 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigAugmentation/fold2'
fold7 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigAugmentation/fold3'
fold8 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/OrigAugmentation/fold4'

train_dataset = BrainSphere(fold3,fold1,fold2, fold7,fold5,fold6)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#conv_type = "RePa"   # "RePa" or "DiNe"
#pooling_type = "mean"  # "max" or "mean" 
model = Unet(2, 35) # UNet or UNet_small or naive_gCNN or UNet_interpolation or SegNet


print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up = 100, step_size_down=100, cycle_momentum=False)


def get_learning_rate(epoch):
    limits = [3, 5, 10, 15, 100]
    lrs = [1,  0.5, 0.1, 0.05, 0.01, 0.001]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

def train_step(data, target):
    model.train()
    data, target = data.cuda(cuda), target.cuda(cuda)

    prediction = model(data)
    
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def compute_dice(pred, gt):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    #dice = 1- len(np.nonzero(gt - pred)[0])/len(pred)
    
    dice = np.zeros(35)    
#    for i in range(len(dice)):
#        gt_indices = np.where(gt == i)[0]
#        pred_indices = np.where(pred == i)[0]
#        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    
    for i in range(len(dice)):
        gt_indices = np.where(gt == i)[0]
        dice[i] = 1- len(np.nonzero(gt[gt_indices] - pred[gt_indices])[0])/len(gt_indices) 
        
    a_dice = 1 - len(np.nonzero(gt - pred)[0])/len(gt)

    return dice, a_dice


def val_during_training(dataloader):
    model.eval()

    dice_all = np.zeros((len(dataloader),35))
    a_dice_all = np.zeros(len(dataloader))
    #dice_all = np.zeros(len(dataloader))
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze(0)
        target = target.squeeze(0)
        data, target = data.cuda(cuda), target.cuda(cuda)
        with torch.no_grad():
            prediction = model(data)
            
        prediction = prediction.max(1)[1]
        #dice_all[batch_idx,:] = compute_dice(prediction, target)
        dice_all[batch_idx,:], a_dice_all[batch_idx] = compute_dice(prediction, target)
   
    return dice_all, a_dice_all


for epoch in range(200):

    train_dice, train_a_dice = val_during_training(train_dataloader)
#    train_dice = np.mean(dice)
    print("train Dice: ", np.mean(train_dice, axis=0))
    print("train_a_dice, mean, std: ", np.mean(train_a_dice), np.std(train_a_dice))
    
    val_dice, val_a_dice = val_during_training(val_dataloader)
#    val_dice = np.mean(dice)
    print("val Dice: ", np.mean(val_dice, axis=0))
    print("val_a_dice, mean, std: ", np.mean(val_a_dice), np.std(val_a_dice))
    writer.add_scalars('data/Dice', {'train': np.mean(train_a_dice), 'val':  np.mean(val_a_dice)}, epoch)
           
    scheduler.step(np.mean(val_a_dice))
#    lr = get_learning_rate(epoch)
#    optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze(0)
        target = target.squeeze(0)
        loss = train_step(data, target)

#        for CyclicLR scheduler
#        print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
#        scheduler.step()

        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
    
        
    
    torch.save(model.state_dict(), os.path.join("NAMIC.pkl"))
    #scheduler.step(train_dice)
