#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:48:12 2018

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
from bscnn.brainSphereCNN import BrainSphereCNN


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
    
    
def compute_dice(pred, gt):

    pred = pred.numpy()
    gt = gt.numpy()
    
    dice = np.zeros(36)
    for i in range(36):
        gt_indices = np.where(gt == i)[0]
        pred_indices = np.where(pred == i)[0]
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return dice
    

learning_rate = 0.1
momentum = 0.9
wd = 0.0001
batch_size = 1
test_dataset_path = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/test'
test_dataset = BrainSphere(test_dataset_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

#    dataiter = iter(test_dataloader)
#    data, target = dataiter.next()

model = BrainSphereCNN(36)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
model.load_state_dict(torch.load('/home/zfq/graphsage-simple/state.pkl'))
model.eval()

dice_all = np.zeros((len(test_dataloader),36))
for batch_idx, (data, target) in enumerate(test_dataloader):
    data = data.squeeze()
    target = target.squeeze()
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        prediction = model(data)
    
    dice_all[batch_idx,:] = compute_dice(prediction, target)

print(np.mean(dice_all, axis=0))
print("average dice:", np.mean(dice_all))


    
    
    
