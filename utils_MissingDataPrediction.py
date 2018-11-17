#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:50:35 2018

@author: fenqiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:29:28 2018

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


from models import *


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
        feats_Y0 = sio.loadmat(file)
        feats_Y0 = feats_Y0['data']
        feats_Y0 = (feats_Y0 - np.tile(np.min(feats_Y0,0), (len(feats_Y0),1)))/(np.tile(np.max(feats_Y0,0), (len(feats_Y0),1)) - np.tile(np.min(feats_Y0,0), (len(feats_Y0),1)))
        feats_Y0 = (feats_Y0 - np.tile(np.mean(feats_Y0, 0), (len(feats_Y0), 1)))*2
        feats_Y1 = sio.loadmat(file[:-3] + '.Y1')
        thickness_Y1 = feats_Y1['data'][:,0]

        return feats_Y0.astype(np.float32), thickness_Y1.astype(np.float32)

    def __len__(self):
        return len(self.files)


batch_size = 1
fold1 = '/media/fenqiang/DATA/unc/Data/MissingDataPrediction/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/MissingDataPrediction/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/MissingDataPrediction/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/MissingDataPrediction/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/MissingDataPrediction/fold5'

train_dataset = BrainSphere(fold1, fold2, fold3, fold4)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

all_data = torch.tensor([])
for batch_idx, (data, target) in enumerate(train_dataloader):
    all_data = torch.cat((all_data, target.squeeze()), 0)

mean = torch.mean(all_data).numpy()
std = torch.std(all_data).numpy()
