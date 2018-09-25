#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:21:19 2018

@author: zfq
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

from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')
from utils import compute_weight

#from gCNN import * 
from models import *
from UNet_interpolation import *
from SegNet import *

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, root2 = None):

        self.files = sorted(glob.glob(os.path.join(root1, '*.mat')))    
        if root2 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root2, '*.mat')))

    def __getitem__(self, index):
        file = self.files[index]
        feats = sio.loadmat(file)
        feats = feats['data']
        feats = (feats - np.tile(np.min(feats,0), (len(feats),1)))/(np.tile(np.max(feats,0), (len(feats),1)) - np.tile(np.min(feats,0), (len(feats),1)))
        feats = feats - np.tile(np.mean(feats, 0), (len(feats), 1))
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        return feats.astype(np.float32), label.astype(np.long)

    def __len__(self):
        return len(self.files)


learning_rate = 0.1
momentum = 0.99
weight_decay = 0.0001
batch_size = 1
fold1 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold1'
fold2 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold2'
fold3 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold3'
train_dataset = BrainSphere(fold2, fold3)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

conv_type = "RePa"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = UNet_small(36, conv_type, pooling_type) # UNet or UNet_small or naive_gCNN or UNet_interpolation or SegNet


print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel')


def train_step(data, target):
    model.train()
    data, target = data.cuda(), target.cuda()

    prediction = model(data)
    
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def compute_dice(pred, gt):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    dice = np.zeros(36)
    for i in range(36):
        gt_indices = np.where(gt == i)[0]
        pred_indices = np.where(pred == i)[0]
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return dice


def val_during_training():
    model.eval()

    dice_all = np.zeros((len(val_dataloader),36))
    for batch_idx, (data, target) in enumerate(val_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(data)
            
        prediction = prediction.max(1)[1]
        dice_all[batch_idx,:] = compute_dice(prediction, target)
    val_dice = np.mean(dice_all)
    print("Val Dice: ", np.mean(dice_all, axis=0))

    return val_dice

def val_train_during_training():
    model.eval()

    dice_all = np.zeros((len(train_dataloader),36))
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(data)
            
        prediction = prediction.max(1)[1]
        dice_all[batch_idx,:] = compute_dice(prediction, target)
    train_dice = np.mean(dice_all)
    print("Train Dice: ", np.mean(dice_all, axis=0))

    return train_dice

a=0
train_dice = [0, 0, 0, 0, 0]
for epoch in range(100):

    for p in optimizer.param_groups:
        print("learning rate = {}".format(p['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        loss = train_step(data, target)

        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)

    val_dice = val_during_training()
    print("Val Dice: ",val_dice)   
    a = val_train_during_training()
    writer.add_scalars('data/Dice', {'train': a, 'val': val_dice}, epoch)

    scheduler.step(a)

    train_dice[epoch % 5] = a
    print("last five train Dice: ",train_dice)
    if np.std(np.array(train_dice)) <= 0.00001:
        torch.save(model.state_dict(), os.path.join("state.pkl"))
        break

    if epoch % 5 == 0:
       # a = val_train_during_training()
        #print("train Dice:", a)
        torch.save(model.state_dict(), os.path.join("state.pkl"))
  
