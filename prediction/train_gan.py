#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:52:30 2018

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


from tensorboardX import SummaryWriter
writer = SummaryWriter('log/gan')

from model import Unet, ResNet

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
        raw_Y0 = raw_Y0['data'][:,[1, 2]]
#        feats_Y0 = (raw_Y0 - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))/(np.tile(np.max(raw_Y0,0), (len(raw_Y0),1)) - np.tile(np.min(raw_Y0,0), (len(raw_Y0),1)))
#        feats_Y0 = feats_Y0 * 2.0 - 1.0
        feats_Y0 = (raw_Y0 - np.tile([0.0081, 1.8586], (len(raw_Y0),1)))/np.tile([0.5169, 0.4395], (len(raw_Y0),1))
        
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        raw_Y1 = raw_Y1['data'][:,2]   # 0: curv, 1: sulc, 2: thickness, 3: x, 4: y, 5: z
#        feats_Y1 = (raw_Y1 - np.tile(np.min(raw_Y1), len(raw_Y1)))/(np.tile(np.max(raw_Y1), len(raw_Y1)) - np.tile(np.min(raw_Y1), len(raw_Y1)))
#        feats_Y1 = feats_Y1 * 2.0 - 1.0
        
        return feats_Y0.astype(np.float32), raw_Y1.astype(np.float32)

    def __len__(self):
        return len(self.files)

cuda = torch.device('cuda:0') 
learning_rate = 0.00005
batch_size = 1
fold1 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold1'
fold2 = '/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed150/fold2'

train_dataset = BrainSphere(fold1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = BrainSphere(fold2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def val_during_training(dataloader):
    generator.eval()

    mae = torch.tensor([])
    mre = torch.tensor([])
    for batch_idx, (data, raw_target) in enumerate(dataloader):
        data = data.squeeze(0).cuda(cuda)
        with torch.no_grad():
            prediction = generator(data)
        prediction = prediction.squeeze().cpu() 
        raw_target = raw_target.squeeze(0).cpu()
#        prediction = (prediction + 1.0) / 2.0 * (raw_target.max() - raw_target.min()) + raw_target.min()
        mae = torch.cat((mae, torch.abs(prediction - raw_target)), 0)
        mre = torch.cat((mre, torch.abs(prediction - raw_target)/raw_target), 0)
        
    m_mae, s_mae = torch.mean(mae), torch.std(mae)
    m_mre, s_mre= torch.mean(mre), torch.std(mre)

    return m_mae, s_mae, m_mre, s_mre


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_learning_rate(epoch):
    limits = [2, 4, 6]
    lrs = [1, 0.1, 0.01, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

# Loss function
adversarial_loss = nn.BCELoss()
mseloss = nn.MSELoss()
l1Loss = nn.L1Loss()
#d_subject_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Unet(2, 1)
discriminator = ResNet(1, 1)

#d_b = D_backbone()
#d_r = D_RealFake()
#d_s = D_Subject(len(train_dataloader))

generator.cuda(cuda)
discriminator.cuda(cuda)
discriminator.load_state_dict(torch.load('/home/fenqiang/Spherical_U-Net/trained_models/resnet_for_real_LRpred.pkl'))
adversarial_loss.cuda(cuda)
mseloss.cuda(cuda)
l1Loss.cuda(cuda)
generator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.00001, momentum=0.99)


# ----------
#  Training
# ----------


#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()

for epoch in range(100):
   
    m_mae, s_mae, m_mre, s_mre  = val_during_training(train_dataloader)
    print("Train: mean mae, std mae, mean mre, std mre : {:.4} {:.4} {:.4} {:.4}".format(m_mae, s_mae, m_mre, s_mre))
    writer.add_scalars('data/mean', {'train': m_mae}, epoch)
    m_mae, s_mae, m_mre, s_mre = val_during_training(val_dataloader)
    print("Val: mean mae, std mae, mean mre, std mre : {:.4} {:.4} {:.4} {:.4}".format(m_mae, s_mae, m_mre, s_mre))    
    writer.add_scalars('data/mean', {'val': m_mae}, epoch)


    for p in optimizer_G.param_groups:
        print("learning rate for G = {}".format(p['lr']))
#    lr = get_learning_rate(epoch)
    for p in optimizer_D.param_groups:
#        p['lr'] = lr
        print("learning rate for D = {}".format(p['lr']))
    
    for i, (data, target) in enumerate(train_dataloader):
       
        generator.train()
        
        data, target = data.squeeze(0).cuda(cuda), target.squeeze(0).cuda(cuda)
        
        # Adversarial ground truths
        valid = torch.ones(1, requires_grad=False).cuda(cuda)
        fake = torch.zeros(1, requires_grad=False).cuda(cuda)

        # ----------------- 
        #  Train Generator and d_backbone
        # -----------------
       
        optimizer_G.zero_grad() 
        
        # Generate a batch of images
        gen_imgs = generator(data)

        # loss measures pixel wise regression
        g_loss_vertex = l1Loss(gen_imgs.squeeze(), target)
        # Loss measures generator's ability to fool the discriminator
        g_loss_ad = adversarial_loss(discriminator(gen_imgs), valid)
        
#        d_r_loss = adversarial_loss(d_r(temp), valid)
#        d_s_loss = d_subject_loss(d_s(temp), index.cuda())
#        g_loss = d_r_loss + d_s_loss
        g_loss = 0.5 * g_loss_vertex + 0.5 * g_loss_ad
        g_loss.backward()
        optimizer_G.step()
#        optimizer_D_B.step()
        
        # ---------------------
        #  Train discriminator
        # ---------------------
    
        optimizer_D.zero_grad()
     
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(target.unsqueeze(1)), torch.FloatTensor(1).uniform_(0.8, 1.0).cuda(cuda))
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), torch.FloatTensor(1).uniform_(0, 0.2).cuda(cuda))
        d_loss = 0.5 * real_loss + 0.5 * fake_loss

        d_loss.backward()
        optimizer_D.step()
        
        # ---------------------
        #  Train d_s
        # ---------------------
#        optimizer_D_S.zero_grad()
#        d_s_loss_raw = d_subject_loss(d_s(temp2.detach()), index.cuda())
#        d_s_loss_gen = d_subject_loss(d_s(temp.detach()), index.cuda())
#        d_s_loss = d_s_loss_raw + d_s_loss_gen
#        d_s_loss.backward()
#        optimizer_D_S.step()
#        

        print ("[Epoch %d] [Batch %d/%d] [D real_loss: %f] [D fake_loss: %f] [G g_loss_vertex: %f] [G g_loss_ad: %f]" % (epoch, i, len(train_dataloader),
                                                            real_loss.item(), fake_loss.item(), g_loss_vertex.item(), g_loss_ad.item()))

        writer.add_scalars('loss', {'G':g_loss , 'D': d_loss}, epoch*len(train_dataloader) + i)
        
    
        

#        batches_done = epoch * len(dataloader) + i
#        if batches_done % opt.sample_interval == 0:
#            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
