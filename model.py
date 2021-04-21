#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import torch
import numpy as np
import torch.nn as nn

from sphericalunet.utils.utils import *
from sphericalunet.layers import *


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
    
    
class Unet_40k(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_40k, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = Get_neighs_order()
        neigh_orders = neigh_orders[1:]
        a, b, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        chs = [in_ch, 32, 64, 128, 256, 512]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
      
        self.up1 = up_block(conv_layer, chs[5], chs[4], neigh_orders[3], upconv_top_index_642, upconv_down_index_642)
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_orders[2], upconv_top_index_2562, upconv_down_index_2562)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[1], upconv_top_index_10242, upconv_down_index_10242)
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_orders[0], upconv_top_index_40962, upconv_down_index_40962)
        
        self.outc = nn.Sequential(
                nn.Linear(chs[1], out_ch)
                )
                
        
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
        
        x = self.outc(x) # 40962 * 36
        return x


class Unet_160k(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(Unet_160k, self).__init__()

        #neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        #neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        
        neigh_orders = Get_neighs_order()
        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        chs = [in_ch, 32, 64, 128, 256, 512]
        
        conv_layer = onering_conv_layer

        self.down1 = down_block(conv_layer, chs[0], chs[1], neigh_orders[0], None, True)
        self.down2 = down_block(conv_layer, chs[1], chs[2], neigh_orders[1], neigh_orders[0])
        self.down3 = down_block(conv_layer, chs[2], chs[3], neigh_orders[2], neigh_orders[1])
        self.down4 = down_block(conv_layer, chs[3], chs[4], neigh_orders[3], neigh_orders[2])
        self.down5 = down_block(conv_layer, chs[4], chs[5], neigh_orders[4], neigh_orders[3])
      
        self.up1 = up_block(conv_layer, chs[5], chs[4], neigh_orders[3], upconv_top_index_2562, upconv_down_index_2562)
        self.up2 = up_block(conv_layer, chs[4], chs[3], neigh_orders[2], upconv_top_index_10242, upconv_down_index_10242)
        self.up3 = up_block(conv_layer, chs[3], chs[2], neigh_orders[1], upconv_top_index_40962, upconv_down_index_40962)
        self.up4 = up_block(conv_layer, chs[2], chs[1], neigh_orders[0], upconv_top_index_163842, upconv_down_index_163842)
        
        self.outc = nn.Sequential(
                nn.Linear(chs[1], out_ch)
                )
                
        
    def forward(self, x):
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2) # 163842 * 32
        
        x = self.outc(x) # 163842 * 36
        return x
