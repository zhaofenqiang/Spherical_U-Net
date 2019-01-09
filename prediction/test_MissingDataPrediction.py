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
        
        return raw_Y0.astype(np.float32), raw_Y1.astype(np.float32), file

    def __len__(self):
        return len(self.files)



cuda = torch.device('cuda:0') 
batch_size = 1

fold_test = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold_test'

test_dataset = BrainSphere(fold_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(test_dataloader)
#    data, target,_,file = dataiter.next()

conv_type = "DiNe"   # "RePa" or "DiNe"
pooling_type = "mean"  # "max" or "mean" 
model = Unet(3, 1)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
model.load_state_dict(torch.load('/home/fenqiang/Spherical_U-Net/missingthickness.pkl'))
model.eval()

mae = torch.tensor([])
mre = torch.tensor([])
for batch_idx, (data, target, file) in enumerate(test_dataloader):
    data = data.squeeze(0).cuda(cuda)
    
    with torch.no_grad():
        prediction = model(data)
    
    prediction = prediction.squeeze().cpu() 
    target = target.squeeze(0)
    mae = torch.cat((mae, torch.abs(prediction - target)), 0)
    mre = torch.cat((mre, torch.abs(prediction - target)/target), 0)
    
    np.savetxt('./pred/unet_' + file[0].split('/')[-1][0:-3] + '_thickness.txt', prediction.cpu().numpy())

m_mae, s_mae = torch.mean(mae), torch.std(mae)
m_mre, s_mre= torch.mean(mre), torch.std(mre)
print("test: mean mae, std mae, mean mre, std mre : {:.4} {:.4} {:.4} {:.4}".format(m_mae, s_mae, m_mre, s_mre))
