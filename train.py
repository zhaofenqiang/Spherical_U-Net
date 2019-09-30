#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:21:19 2018

@author: zfq
"""

import torch
import torch.nn as nn
import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os

from utils import compute_weight
from comparison_model import *
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
batch_size = 1
model_name = 'Unet_infant'  # 'Unet_infant', 'Unet_18', 'Unet_2ring', 'Unet_repa', 'fcn', 'SegNet', 'SegNet_max'
up_layer = 'upsample_interpolation' # 'upsample_interpolation', 'upsample_fixindex' 
in_channels = 3
out_channels = 36
learning_rate = 0.001
momentum = 0.99
weight_decay = 0.0001
fold = 1 # 1,2,3 
################################################################


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


fold1 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold2'
fold3 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold3'
fold4 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold4'
fold5 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold5'
fold6 = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold6'

if fold == 1:
    train_dataset = BrainSphere(fold3,fold6,fold2,fold5)          
    val_dataset = BrainSphere(fold1)
elif fold == 2:
    train_dataset = BrainSphere(fold1,fold4,fold3,fold6)          
    val_dataset = BrainSphere(fold2)
elif fold == 3:
    train_dataset = BrainSphere(fold1,fold4,fold2,fold5)          
    val_dataset = BrainSphere(fold3)
else:
    raise NotImplementedError('fold name is wrong!')
    
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


if model_name == 'Unet_infant':
    model = Unet_infant(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'Unet_18':
     model = Unet_18(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'Unet_2ring':
    model = Unet_2ring(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'Unet_repa':
    model = Unet_repa(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'fcn':
    model = fcn(in_ch=in_channels, out_ch=out_channels)
elif model_name == 'SegNet':
    model = SegNet(in_ch=in_channels, out_ch=out_channels, up_layer=up_layer)
elif model_name == 'SegNet_max':
    model = SegNet_max(in_ch=in_channels, out_ch=out_channels)
else:
    raise NotImplementedError('model name is wrong!')

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)


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


train_dice = [0, 0, 0, 0, 0]
for epoch in range(100):
    
    train_dc = val_during_training(train_dataloader)
    print("train Dice: ", np.mean(train_dc, axis=0))
    print("train_dice, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, 1)))
    
    val_dc = val_during_training(val_dataloader)
    print("val Dice: ", np.mean(val_dc, axis=0))
    print("val_dice, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, 1)))
    writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)    

    scheduler.step(np.mean(val_dc))

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

    train_dice[epoch % 5] = np.mean(train_dc)
    print("last five train Dice: ",train_dice)
    if np.std(np.array(train_dice)) <= 0.00001:
        torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+"_final.pkl"))
        break

    torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+".pkl"))