#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:38:21 2018

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
        raw_Y0 = raw_Y0['data'][:,[3, 4, 5]]
        feats_Y0 = (raw_Y0 - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))/(np.tile(np.max(raw_Y0,0), (len(raw_Y0),1)) - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))
        feats_Y0 = feats_Y0 * 2.0 - 1.0
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        raw_Y1 = raw_Y1['data'][:,[3,4,5]]   # 0: curv, 1: sulc, 2: thickness, 3: x, 4: y, 5: z
        feats_Y1 = (raw_Y1 - np.tile(np.min(raw_Y1,0), (len(raw_Y1),1)))/(np.tile(np.max(raw_Y1,0), (len(raw_Y1),1)) - np.tile(np.min(raw_Y1,0), (len(raw_Y1),1)))
        feats_Y1 = feats_Y1 * 2.0 - 1.0
        
        return feats_Y0.astype(np.float32), feats_Y1.astype(np.float32), raw_Y1.astype(np.float32), file

    def __len__(self):
        return len(self.files)



cuda = torch.device('cuda:0') 
batch_size = 1
fold1 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold2'

test_dataset = BrainSphere(fold2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(test_dataloader)
#    data, target,raw_Y1,file = dataiter.next()

conv_type = "DiNe"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = Unet(3,3)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
model.load_state_dict(torch.load('/home/fenqiang/Spherical_U-Net/pred_pos.pkl'))
model.eval()

all_error = np.zeros((len(test_dataloader)*40962*3))
mre = np.zeros((len(test_dataloader)*40962*3))
for batch_idx, (data, _, raw_Y1, file) in enumerate(test_dataloader):
    
    data = data.squeeze(0).cuda(cuda)
    with torch.no_grad():
        prediction = model(data)
        
    prediction = prediction.squeeze().cpu().detach().numpy()
    raw_Y1 = raw_Y1.squeeze().numpy()
    prediction = (prediction + 1.0) / 2.0 * (np.tile(np.max(raw_Y1,0), (len(raw_Y1),1)) - np.tile(np.min(raw_Y1,0), (len(raw_Y1),1))) + np.tile(np.min(raw_Y1,0), (len(raw_Y1),1))
    
    np.savetxt('./pred/unet_' + file[0].split('/')[-1][0:-3] + '_pos.txt', prediction) 
    
    all_error[batch_idx*40962*3 : (batch_idx + 1)*40962*3] = np.absolute(prediction - raw_Y1).flatten()
    mre[batch_idx*40962*3 : (batch_idx + 1)*40962*3] = (np.absolute(prediction - raw_Y1)/raw_Y1).flatten()
      

mean, std = np.mean(all_error), np.std(all_error)
m_mre, s_mre = np.mean(mre), np.std(mre)
print(mean, std, m_mre, s_mre)









