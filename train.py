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

from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')
from utils import compute_weight

#from gCNN import * 
from comparison_model import *
#from UNet_interpolation import *
#from SegNet import *

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
        
        feats = data[:,[0,1,2]]
        feat_max = np.max(feats,0)
        for i in range(np.shape(feats)[1]):
            feats[:,i] = feats[:, i]/feat_max[i]
	
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        return feats.astype(np.float32), label.astype(np.long)

    def __len__(self):
        return len(self.files)


learning_rate = 0.001
momentum = 0.99
weight_decay = 0.0001
batch_size = 1
cuda = torch.device('cuda:0')

fold1 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold5'
fold6 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold6'

train_dataset = BrainSphere(fold3, fold2)          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Modify it !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold1)                  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Modify it !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = SegNet_max(3, 36)                         # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Modify it !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    
    dice = np.zeros(36)
    for i in range(36):
        gt_indices = np.where(gt == i)[0]
        pred_indices = np.where(pred == i)[0]
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return dice


def val_during_training(dataloader):
    model.eval()

    dice_all = np.zeros((len(dataloader),36))
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze()
        target = target.squeeze()
        data, target = data.cuda(cuda), target.cuda(cuda)
        with torch.no_grad():
            prediction = model(data)
            
        prediction = prediction.max(1)[1]
        dice_all[batch_idx,:] = compute_dice(prediction, target)

    return dice_all

#
#a=0
#train_dice = [0, 0, 0, 0, 0]
for epoch in range(100):
    
    train_dice = val_during_training(train_dataloader)
#    train_dice = np.mean(dice)
    print("train Dice: ", np.mean(train_dice, axis=0))
    print("train_dice, mean, std: ", np.mean(train_dice), np.std(np.mean(train_dice, 1)))
    
    val_dice = val_during_training(val_dataloader)
#    val_dice = np.mean(dice)
    print("val Dice: ", np.mean(val_dice, axis=0))
    print("val_dice, mean, std: ", np.mean(val_dice), np.std(np.mean(val_dice, 1)))
    writer.add_scalars('data/Dice', {'train': np.mean(train_dice), 'val':  np.mean(val_dice)}, epoch)    

    scheduler.step(np.mean(val_dice))
#    lr = get_learning_rate(epoch)
#    optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        loss = train_step(data, target)

        print("[{}:{}/{}]  LOSS={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)

#    train_dice[epoch % 5] = a
#    print("last five train Dice: ",train_dice)
#    if np.std(np.array(train_dice)) <= 0.00001:
#        torch.save(model.state_dict(), os.path.join("noenate.pkl"))
#        break
#
#    if epoch % 5 == 0:
#       # a = val_train_during_training()
#        #print("train Dice:", a)
        
    torch.save(model.state_dict(), os.path.join("neonate_SegNet_max.pkl"))        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Modify it !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!