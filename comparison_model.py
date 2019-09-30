#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: zfq

This is for brain parcellation. 
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from utils import *
from layers import *

class naive_16conv(nn.Module):
    """Define the baseline architecture of 16 convolutional layers
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            
    Input: 
        N x in_ch,tensor
    Return:
        N x out_ch, tensor
    """    

    def __init__(self, in_ch, out_ch, conv_type='one_ring'):
        super(naive_16conv, self).__init__()
        
        neigh_orders = Get_neighs_order()
       
        if conv_type == "one_ring":
            conv_layer = onering_conv_layer
        if conv_type == "two_ring":
            conv_layer = tworing_conv_layer
        if conv_type == "repa":
            conv_layer = repa_conv_layer
                
        sequence = []
        sequence.append(conv_layer(in_ch, 64, neigh_orders[1], None, None))
        sequence.append(nn.BatchNorm1d(64, momentum=0.15, affine=True, track_running_stats=False))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))

        for l in range(0, 14):
            sequence.append(conv_layer(64, 64, neigh_orders[1], None, None))
            sequence.append(nn.BatchNorm1d(64, momentum=0.15, affine=True, track_running_stats=False))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))
 
        sequence.append(conv_layer(64, out_ch, neigh_orders[1], None, None))

        self.sequential = nn.Sequential(*sequence)
      
    def forward(self, x):
        x = self.sequential(x)    
        return x


class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()


#        Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x
    

class Unet_infant(nn.Module):
    """Define the Spherical UNet structure for spherical cortical surface datra

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_infant, self).__init__()

        neigh_orders = Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        chs = [in_ch, 64, 128, 256, 512, 1024]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[5], neigh_orders[4])
      
        self.up1 = up_block(conv_layer, chs[5], chs[4], neigh_orders[4], upconv_top_index_162, upconv_down_index_162)
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        
#        self.dp = nn.Dropout(p=0.4)
        self.outc = nn.Sequential(
#                conv_layer(chs[1], chs[1], neigh_orders_40962),
#                nn.BatchNorm1d(chs[1]),
#                nn.LeakyReLU(0.2),
#                conv_layer(chs[1], chs[1], neigh_orders_40962),
#                nn.BatchNorm1d(chs[1]),
#                nn.LeakyReLU(0.2),
                nn.Linear(chs[1], out_ch)
                )
                
        
#        self.tanh = nn.Tanh()

    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2) # 40962 * 32
        
#        x = self.dp(x) #40962 * 32
        x = self.outc(x) # 40962 * 35
#        x = self.tanh(x)
        return x



class Unet_18(nn.Module):
    """Define the UNet18 structure for spherical cortical surface datra

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_18, self).__init__()

        neigh_orders = Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        chs = [in_ch, 32, 64, 128, 256]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[1], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[2], neigh_orders[1])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[3], neigh_orders[2])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[4], neigh_orders[3])
      
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        
        self.outc = nn.Linear(chs[1], out_ch)

    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up2(x5, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2) # 40962 * 32
        
        x = self.outc(x) # 40962 * 35
        return x    
    
    
class Unet_2ring(nn.Module):
    """Define the Spherical UNet using 2ring filter for spherical surface data

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_2ring, self).__init__()

        neigh_orders = Get_neighs_order()
        neigh_2ring_orders = Get_2ring_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        chs = [in_ch, 64, 128, 256, 512, 1024]
        
        conv_layer = tworing_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_2ring_orders[1], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_2ring_orders[2], neigh_orders[1])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_2ring_orders[3], neigh_orders[2])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_2ring_orders[4], neigh_orders[3])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_2ring_orders[5], neigh_orders[4])
      
        self.up1 = up_block(conv_layer, chs[5], chs[4], neigh_2ring_orders[4], upconv_top_index_162, upconv_down_index_162)
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_2ring_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_2ring_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_2ring_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        
        self.outc = nn.Linear(chs[1], out_ch)

    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2) # 40962 * 32
        
        x = self.outc(x) # 40962 * 35
        return x

    
    
class Unet_repa(nn.Module):
    """The Spherical UNet architecture using rectangular patch

    """  

    def __init__(self, in_ch, out_ch):
        super(Unet_repa, self).__init__()
        
        neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        neigh_orders = Get_neighs_order()
        weight_10242, weight_2562, weight_642, weight_162, weight_42 = Get_weights()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        self.feature_dims = [in_ch, 64, 128, 256, 512, 1024]

        conv_layer = repa_conv_layer
        
        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_indices_10242, weight_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[1], neigh_indices_10242, weight_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool1 = pool_layer(neigh_orders[1])
        self.conv2_1 = conv_layer(self.feature_dims[1], self.feature_dims[2],  neigh_indices_2562, weight_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv2_2 = conv_layer(self.feature_dims[2], self.feature_dims[2],  neigh_indices_2562, weight_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool2 = pool_layer(neigh_orders[2])
        self.conv3_1 = conv_layer(self.feature_dims[2], self.feature_dims[3],neigh_indices_642, weight_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv3_2 = conv_layer(self.feature_dims[3], self.feature_dims[3],neigh_indices_642, weight_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool3 = pool_layer(neigh_orders[3])
        self.conv4_1 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_indices_162, weight_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv4_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_indices_162, weight_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool4 = pool_layer(neigh_orders[4])
        self.conv5_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_indices_42, weight_42)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv5_2 = conv_layer(self.feature_dims[5], self.feature_dims[5], neigh_indices_42, weight_42)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        
        self.upconv42_162 = upconv_layer( self.feature_dims[5], self.feature_dims[4], upconv_top_index_162, upconv_down_index_162)
        self.conv6_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4], neigh_indices_162, weight_162)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv6_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_indices_162, weight_162)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.upconv162_642 = upconv_layer(self.feature_dims[4], self.feature_dims[3], upconv_top_index_642, upconv_down_index_642)
        self.conv7_1 = conv_layer(self.feature_dims[3] * 2, self.feature_dims[3], neigh_indices_642, weight_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv7_2 = conv_layer(self.feature_dims[3], self.feature_dims[3], neigh_indices_642, weight_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.upconv642_2562 = upconv_layer(self.feature_dims[3], self.feature_dims[2], upconv_top_index_2562, upconv_down_index_2562)
        self.conv8_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2], neigh_indices_2562, weight_2562)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv8_2 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_indices_2562, weight_2562)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.upconv2562_10242 = upconv_layer(self.feature_dims[2], self.feature_dims[1], upconv_top_index_10242, upconv_down_index_10242)
        self.conv9_1 = conv_layer(self.feature_dims[1] * 2, self.feature_dims[1],  neigh_indices_10242, weight_10242)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv9_2 = conv_layer(self.feature_dims[1], self.feature_dims[1],  neigh_indices_10242, weight_10242)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.conv10 = conv_layer(self.feature_dims[1], out_ch, neigh_indices_10242, weight_10242)
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu(self.bn1_2(self.conv1_2(x)))
        
        x = self.pool1(x1)
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x)))
        
        x = self.pool2(x2)          
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x3 = self.relu(self.bn3_2(self.conv3_2(x)))
        
        x = self.pool3(x3)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x4 = self.relu(self.bn4_2(self.conv4_2(x)))
        
        x = self.pool4(x4)
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        
        x = self.upconv42_162(x)  # 162* 512
        x = torch.cat((x, x4), 1) # 162 * 1024
        x = self.relu(self.bn6_1(self.conv6_1(x))) 
        x = self.relu(self.bn6_2(self.conv6_2(x)))  # 162 * 512

        x = self.upconv162_642(x)  # 642* 256
        x = torch.cat((x, x3), 1) # 642 * 512
        x = self.relu(self.bn7_1(self.conv7_1(x))) 
        x = self.relu(self.bn7_2(self.conv7_2(x)))  # 642 * 256 

        x = self.upconv642_2562(x)  # 2562* 128
        x = torch.cat((x, x2), 1) # 2562 * 256
        x = self.relu(self.bn8_1(self.conv8_1(x))) 
        x = self.relu(self.bn8_2(self.conv8_2(x)))  # 2562 * 128 
        
        x = self.upconv2562_10242(x)  # 10242* 64
        x = torch.cat((x, x1), 1) # 10242 * 128
        x = self.relu(self.bn9_1(self.conv9_1(x))) 
        x = self.relu(self.bn9_2(self.conv9_2(x)))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x
    
    
class fcn(nn.Module):
    """Define the FCN architecture for icosaherdron discretized spherical surface data

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(fcn, self).__init__()

        neigh_orders = Get_neighs_order()
        upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 
        upsample_neighs_10242, upsample_neighs_2562, upsample_neighs_642, upsample_neighs_162, upsample_neighs_42 = Get_upsample_neighs_order()
        
        conv_layer = onering_conv_layer

        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(3, 64, neigh_orders[1])
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv1_2 = conv_layer(64, 64, neigh_orders[1])
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool1 = pool_layer(neigh_orders[1])
        self.conv2_1 = conv_layer(64, 128, neigh_orders[2])
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv2_2 = conv_layer(128, 128, neigh_orders[2])
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool2 = pool_layer(neigh_orders[2])
        self.conv3_1 = conv_layer(128, 256, neigh_orders[3])
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv3_2 = conv_layer(256, 256, neigh_orders[3])
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool3 = pool_layer(neigh_orders[3])
        self.conv4_1 = conv_layer(256, 512, neigh_orders[4])
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv4_2 = conv_layer(512, 512, neigh_orders[4])
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        self.pool4 = pool_layer(neigh_orders[4])
        self.conv5_1 = conv_layer(512, 1024, neigh_orders[5])
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        self.conv5_2 = conv_layer(1024, 1024, neigh_orders[5])
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats, momentum=0.15, affine=True, track_running_stats=False)
        
        
        self.up42_2562 = nn.Sequential(
                 upsample_interpolation(upsample_neighs_162),
                 upsample_interpolation(upsample_neighs_642),
                 upsample_interpolation(upsample_neighs_2562)
                )
        
        self.up162_2562 = nn.Sequential(
                 upsample_interpolation(upsample_neighs_642),
                 upsample_interpolation(upsample_neighs_2562)
                )
        
        self.up642_2562 = upsample_interpolation(upsample_neighs_2562)
        
        self.up = upconv_layer(1024+512+256, out_ch, upconv_top_index_10242, upconv_down_index_10242)
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu(self.bn1_2(self.conv1_2(x))) # 10242*64
        
        x = self.pool1(x1)
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x))) # 2562 * 128
        
        x = self.pool2(x2)          
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x3 = self.relu(self.bn3_2(self.conv3_2(x))) # 642*256
        
        x = self.pool3(x3)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x4 = self.relu(self.bn4_2(self.conv4_2(x))) # 162*512
        
        x = self.pool4(x4)
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x5 = self.relu(self.bn5_2(self.conv5_2(x))) # 42 * 1024
        
        x5 = self.up42_2562(x5)  # 2562* 1024
        x4 = self.up162_2562(x4) # 2562*512
        x3 = self.up642_2562(x3) # 2562*256

        x = torch.cat((x5, x4, x3), 1) # 2562 * 1792
        x = self.up(x)
        
        return x


class SegNet(nn.Module):
    """
    The segnet architecture for icosahedron discretized spherical surface data
    
    Parameters:
        up_layer: upsample_interpolation or upsample_fixindex
    """
    def __init__(self, in_ch, out_ch, up_layer):
        super(SegNet, self).__init__()
        
        neigh_orders = Get_neighs_order()
        upsample_neighs_10242, upsample_neighs_2562, upsample_neighs_642, upsample_neighs_162, upsample_neighs_42 = Get_upsample_neighs_order()

        self.feature_dims = [in_ch, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]

        conv_layer = onering_conv_layer
        if up_layer == 'upsample_interpolation':
            up_layer = upsample_interpolation
        if up_layer == 'upsample_fixindex':
            up_layer = upsample_fixindex
                
        
        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders[1])
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders[1])
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats)
        
        self.pool1 = pool_layer(neigh_orders[1])
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders[2])
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders[2])
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        
        self.pool2 = pool_layer(neigh_orders[2])
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders[3])
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders[3])
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        
        self.pool3 = pool_layer(neigh_orders[3])
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders[4])
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders[4])
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        
        self.pool4 = pool_layer(neigh_orders[4])
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[9], neigh_orders[5])
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[9], self.feature_dims[10], neigh_orders[5])
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        
        self.upconv42_162 = up_layer(upsample_neighs_162)
        self.conv6_1 = conv_layer(self.feature_dims[8] * 2, self.feature_dims[8],  neigh_orders[4])
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[8], self.feature_dims[8],  neigh_orders[4])
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)
        
        self.upconv162_642 = up_layer(upsample_neighs_642)
        self.conv7_1 = conv_layer(self.feature_dims[6] * 2, self.feature_dims[6],  neigh_orders[3])
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[6], self.feature_dims[6],  neigh_orders[3])
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.upconv642_2562 = up_layer(upsample_neighs_2562)
        self.conv8_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4],  neigh_orders[2])
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[4], self.feature_dims[4],  neigh_orders[2])
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.upconv2562_10242 = up_layer(upsample_neighs_10242)
        self.conv9_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2],  neigh_orders[1])
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[2],  neigh_orders[1])
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[2], out_ch, neigh_orders[1])
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        
        x = self.pool1(x)
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        
        x = self.pool2(x)          
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        
        x = self.pool3(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        
        x = self.pool4(x)
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        
        x = self.upconv42_162(x)  # 162* 1024
        x = self.relu(self.bn6_1(self.conv6_1(x))) 
        x = self.relu(self.bn6_2(self.conv6_2(x)))  # 162 * 512

        x = self.upconv162_642(x)  # 642* 512
        x = self.relu(self.bn7_1(self.conv7_1(x))) 
        x = self.relu(self.bn7_2(self.conv7_2(x)))  # 642 * 256 

        x = self.upconv642_2562(x)  # 2562* 256
        x = self.relu(self.bn8_1(self.conv8_1(x)))
        x = self.relu(self.bn8_2(self.conv8_2(x)))  # 2562 * 128 
        
        x = self.upconv2562_10242(x)  # 10242* 128
        x = self.relu(self.bn9_1(self.conv9_1(x))) 
        x = self.relu(self.bn9_2(self.conv9_2(x)))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x
    




class SegNet_max(nn.Module):
    """
    SegNet architecture with upsampling using maxpooling indices for spherical data
    """

    def __init__(self, in_ch, out_ch):
        super(SegNet_max, self).__init__()
        
        neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()

        self.feature_dims = [in_ch, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]

        conv_layer = onering_conv_layer
        unpool_layer = upsample_maxindex
        pooling_type = "max"

        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats) 
        self.pool1 = pool_layer(neigh_orders_10242, pooling_type)
        
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        self.pool2 = pool_layer(neigh_orders_2562, pooling_type)
        
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        self.pool3 = pool_layer(neigh_orders_642, pooling_type)
        
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        self.pool4 = pool_layer(neigh_orders_162, pooling_type)
        
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[9], neigh_orders_42)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[9], self.feature_dims[8], neigh_orders_42)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        
        self.unpool1 = unpool_layer(162, neigh_orders_162)
        self.conv6_1 = conv_layer(self.feature_dims[8], self.feature_dims[7], neigh_orders_162)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[7], self.feature_dims[6], neigh_orders_162)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)

        self.unpool2 = unpool_layer(642, neigh_orders_642) 
        self.conv7_1 = conv_layer(self.feature_dims[6], self.feature_dims[5], neigh_orders_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[5], self.feature_dims[4], neigh_orders_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.unpool3 = unpool_layer(2562, neigh_orders_2562) 
        self.conv8_1 = conv_layer(self.feature_dims[4], self.feature_dims[3], neigh_orders_2562)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[3], self.feature_dims[2], neigh_orders_2562)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.unpool4 = unpool_layer(10242, neigh_orders_10242) 
        self.conv9_1 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_orders_10242)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[1], neigh_orders_10242)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[1], out_ch, neigh_orders_10242)
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x))) # 10242 * 64
        x = self.relu(self.bn1_2(self.conv1_2(x))) # 10242 * 64
        x, max_index1 = self.pool1(x) # 2562 * 64
        
        x = self.relu(self.bn2_1(self.conv2_1(x))) # 2562 * 128
        x = self.relu(self.bn2_2(self.conv2_2(x))) # 2562 * 128
        x, max_index2 = self.pool2(x)  # 642 * 128
        
        x = self.relu(self.bn3_1(self.conv3_1(x))) # 642 * 256
        x = self.relu(self.bn3_2(self.conv3_2(x))) #  642 * 256
        x, max_index3 = self.pool3(x) # 162 *256
        
        x = self.relu(self.bn4_1(self.conv4_1(x))) # 162 * 512
        x = self.relu(self.bn4_2(self.conv4_2(x))) # 162 * 512
        x, max_index4 = self.pool4(x) # 42 * 512
        
        x = self.relu(self.bn5_1(self.conv5_1(x))) # 42 * 512
        x = self.relu(self.bn5_2(self.conv5_2(x))) # 42 * 512
     
        x = self.unpool1(x, max_index4)  # 162* 512
        x = self.relu(self.bn6_1(self.conv6_1(x))) 
        x = self.relu(self.bn6_2(self.conv6_2(x)))  # 162 * 512

        x = self.unpool2(x, max_index3)  # 642 * 512
        x = self.relu(self.bn7_1(self.conv7_1(x))) 
        x = self.relu(self.bn7_2(self.conv7_2(x)))  # 642 * 256 

        x = self.unpool3(x, max_index2)  # 2562 * 128
        x = self.relu(self.bn8_1(self.conv8_1(x))) 
        x = self.relu(self.bn8_2(self.conv8_2(x)))  # 2562 * 64 
        
        x = self.unpool4(x, max_index1)  # 10242 * 64
        x = self.relu(self.bn9_1(self.conv9_1(x))) 
        x = self.relu(self.bn9_2(self.conv9_2(x)))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x

