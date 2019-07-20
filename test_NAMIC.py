#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

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
import time

from model import Unet

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1, root2 = None, root3 = None):

        self.files = sorted(glob.glob(os.path.join(root1, '*.mat')))    
        if root2 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root2, '*.mat')))
        if root3 is not None:
            self.files = self.files + sorted(glob.glob(os.path.join(root3, '*.mat')))
                

    def __getitem__(self, index):
        file = self.files[index]
        data = sio.loadmat(file)
        data = data['data']
        
        feats = data[:,[0,1]]
        #feats = feats/np.max(feats)
        #feats = (feats - np.tile(np.min(feats,0), (len(feats),1)))/(np.tile(np.max(feats,0), (len(feats),1)) - np.tile(np.min(feats,0), (len(feats),1)))
        #feats = feats - np.tile(np.mean(feats, 0), (len(feats), 1))
        
        label = data[:,5]-1
        return feats.astype(np.float32), label.astype(np.long), file

    def __len__(self):
        return len(self.files)


batch_size = 1
fold1 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/NAMIC/format_data/fold4'
test_dataset = BrainSphere(fold4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(test_dataloader)
#    data, target = dataiter.next()

model = Unet(2, 35) # UNet or UNet_small or naive_gCNN or UNet_interpolation or SegNet


print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda()
model.load_state_dict(torch.load('/home/fenqiang/Spherical_U-Net/NAMIC.pkl'))
model.eval()


def compute_dice(pred, gt):

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    #dice = 1- len(np.nonzero(gt - pred)[0])/len(pred)
    
#    dice = np.zeros(35)    
#    for i in range(len(dice)):
#        gt_indices = np.where(gt == i)[0]
#        pred_indices = np.where(pred == i)[0]
#        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    
    dice = np.zeros(35)    
    for i in range(len(dice)):
        gt_indices = np.where(gt == i)[0]
        dice[i] = 1- len(np.nonzero(gt[gt_indices] - pred[gt_indices])[0])/len(gt_indices) 

    return dice


dice_all = np.zeros((len(test_dataloader),35))
times = []
for batch_idx, (data, target, file) in enumerate(test_dataloader):
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
    pred = pred.cpu().numpy()
    np.savetxt('/media/fenqiang/DATA/unc/Data/NAMIC/prediction/' + file[0].split('/')[-1].split('.')[0] + '.txt', pred) 

print(np.mean(dice_all, axis=1))
print("average dice:", np.mean(dice_all))
print("average time for one inference:",np.mean(np.array(times)))

