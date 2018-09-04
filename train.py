#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:21:19 2018

@author: zfq

This is for brain parcellation. Implemente the gCNN method in paper "Geometric
 Convolutional Neural Network for Analyzing Surface-Based Neuroimaging Data"
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
from gCNN import gCNN_with_pool
from tensorboardX import SummaryWriter

writer = SummaryWriter('log/with_pool')

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(self.root, '*.mat')))    
        self.transform = transform
        
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
wd = 0.0001
batch_size = 1
train_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/train'
val_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/val'
train_dataset = BrainSphere(train_dataset_path)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(val_dataset_path)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


model = gCNN_with_pool(36)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=momentum, weight_decay=wd)
criterion = nn.CrossEntropyLoss()


def get_learning_rate(epoch):
    limits = [3, 5, 8, 15, 20]
    lrs = [1, 0.5, 0.2, 0.1, 0.01, 0.001]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


def train_step(data, target):
    model.train()
    data, target = data.cuda(), target.cuda()

    prediction = model(data)
    
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dice = np.mean(compute_dice(prediction, target))
    return loss.item(), dice


def compute_dice(pred, gt):

    pred = pred.detach().cpu().numpy()
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



for epoch in range(300):

    val_dice = val_during_training()
    print("Val Dice: ",val_dice)
    if epoch % 5 == 0:
        train_dice = val_train_during_training()
        print("train Dice: ",train_dice)

    writer.add_scalars('data/Dice', {'train': train_dice, 'val': val_dice}, epoch)

    lr = get_learning_rate(epoch)
    print("learning rate = {} and batch size = {}".format(lr, train_dataloader.batch_size))
    for p in optimizer.param_groups:
        #print(p)
        p['lr'] = lr

    total_loss = 0
    total_correct = 0
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        target = target.squeeze()
        loss, dice = train_step(data, target)

        print("[{}:{}/{}]  LOSS={:.4}  DICE={:.4}".format(epoch, 
              batch_idx, len(train_dataloader), loss, dice))
        
        writer.add_scalar('Train/Loss', loss, epoch*70 + batch_idx)
        
    #if epoch % 5 == 0 :
     #   torch.save(model.state_dict(), os.path.join("state.pkl"))

