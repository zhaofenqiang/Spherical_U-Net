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
import time

from sklearn.metrics import f1_score

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
    
    
def compute_dice(pred, gt):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    dice = np.zeros(36)
    for i in range(36):
        gt_indices = np.where(gt == i)[0]
        pred_indices = np.where(pred == i)[0]
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return dice
    

batch_size = 1
fold1 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold1'
fold2 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold2'
fold3 = '/media/zfq/WinE/unc/zhengwang/dataset/format_dataset/90/fold3'

test_dataset = BrainSphere(fold2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(test_dataloader)
#    data, target = dataiter.next()

conv_type = "DiNe"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = UNet(36, conv_type, pooling_type) # UNet or UNet_small or naive_gCNN or UNet_interpolation or SegNet


print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
model.load_state_dict(torch.load('/home/zfq/gCNN/trained_models/UNet_DiNe.pkl'))
model.eval()

dice_all = np.zeros((len(test_dataloader),36))
times = []
for batch_idx, (data, target) in enumerate(test_dataloader):
    data = data.squeeze()
    target = target.squeeze()
    data, target = data.cuda(), target.cuda()
    
    time_start=time.time()    
    with torch.no_grad():
        prediction = model(data)
    time_end=time.time()
    times.append(time_end - time_start)
    
    pred = prediction.max(1)[1]
    dice_all[batch_idx,:] = compute_dice(pred, target)
    pred = prediction.cpu().numpy()
    np.savetxt('/media/zfq/WinE/unc/zhengwang/results/softmax/UNet/UNet_' + str(batch_idx+31) + '.txt', pred) 

print(np.mean(dice_all, axis=1))
print("average dice:", np.mean(dice_all))
print("average time for one inference:",np.mean(np.array(times)))

