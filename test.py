#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:48:12 2018

@author: zfq
"""

import torch

import scipy.io as sio 
import numpy as np
import glob
import os
import time

from comparison_model import *

################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
batch_size = 1
model_name = 'Unet_infant'  # 'Unet_infant', 'Unet_18', 'Unet_2ring', 'Unet_repa', 'fcn', 'SegNet', 'SegNet_max'
up_layer = 'upsample_interpolation' # 'upsample_interpolation', 'upsample_fixindex' 
in_channels = 3
out_channels = 36
fold = 1 # 1,2,3

###############################################################

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
        return feats.astype(np.float32), label.astype(np.long), file

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
    
if fold == 1:
    folder = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold1'
elif fold == 2:
    folder = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold2'
elif fold == 3:
    folder = '/media/fenqiang/DATA/unc/Data/NeonateParcellation/format_dataset/90/fold3'
else:
    raise NotImplementedError('model name is wrong!')

test_dataset = BrainSphere(folder)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

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
model.cuda()
model.load_state_dict(torch.load('/home/fenqiang/Spherical_U-Net/trained_models/'+model_name+'_'+str(fold)+".pkl")) 
model.eval()

dice_all = np.zeros((len(test_dataloader),36))
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
#    np.savetxt('/media/fenqiang/DATA/unc/Data/NeonateParcellation/result/'+model_name +'/' + file[0].split('/')[-1].split('.')[0] + '.txt', pred)                                                  

print('every subject mean dice: ', np.mean(dice_all, axis=1))
print('every roi mean dice: ', np.mean(dice_all, axis=0))
print("all mean dice", np.mean(dice_all))
print("average time for one inference:",np.mean(np.array(times)))

