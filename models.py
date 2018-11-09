#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: zfq

This is for brain parcellation. Implement the geometric U-Net 
and modify it for brain segmentation
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from utils import *
from layers import *

class naive_gCNN(nn.Module):

    def __init__(self, num_classes, conv_type, pooling_type):
        super(naive_gCNN, self).__init__()
        
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
       
        self.feature_dims = [4, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
                
        if conv_type == "RePa":
            conv_layer = gCNN_conv_layer
        if conv_type == "DiNe":
            conv_layer = DiNe_conv_layer
        
        sequence = []

        for l in range(0, len(self.feature_dims) - 2):
            nfeature_in = self.feature_dims[l]
            nfeature_out = self.feature_dims[l + 1]

            sequence.append(conv_layer(nfeature_in, nfeature_out, neigh_orders_10242, None, None))
            sequence.append(nn.BatchNorm1d(nfeature_out))
            sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)
      
        self.out_layer = conv_layer(self.feature_dims[-1], num_classes, neigh_orders_10242, None, None)
        

    def forward(self, x):
        x = self.sequential(x)    
        x = self.out_layer(x)
        return x


class UNet_small(nn.Module):

    def __init__(self, num_classes, conv_type, pooling_type):
        super(UNet_small, self).__init__()
        
        neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        weight_10242, weight_2562, weight_642, weight_162, weight_42 = Get_weights()
        upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        self.feature_dims = [4, 32, 32, 64, 64, 128, 128, 256, 256]
        
        if conv_type == "RePa":
            conv_layer = gCNN_conv_layer
        if conv_type == "DiNe":
            conv_layer = DiNe_conv_layer

        self.relu = nn.Tanh()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats)
        
        self.pool1 = pool_layer(2562, neigh_orders_10242, pooling_type)
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        
        self.pool2 = pool_layer(642, neigh_orders_2562, pooling_type)
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        
        self.pool3 = pool_layer(162, neigh_orders_642, pooling_type)
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        

        self.upconv162_642 = upconv_layer(642, self.feature_dims[7], self.feature_dims[6], upconv_top_index_642, upconv_down_index_642)
        self.conv7_1 = conv_layer(self.feature_dims[6] * 2, self.feature_dims[6], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[6], self.feature_dims[6], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.upconv642_2562 = upconv_layer(2562, self.feature_dims[5], self.feature_dims[4], upconv_top_index_2562, upconv_down_index_2562)
        self.conv8_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.upconv2562_10242 = upconv_layer(10242, self.feature_dims[4], self.feature_dims[2], upconv_top_index_10242, upconv_down_index_10242)
        self.conv9_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[2], num_classes, neigh_orders_10242, neigh_indices_10242, weight_10242)
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu(self.bn1_2(self.conv1_2(x)))
        
        x = self.pool1(x1)                  # 2562*64
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x)))
        
        x = self.pool2(x2)               # 642*128
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x3 = self.relu(self.bn3_2(self.conv3_2(x)))
        
        x = self.pool3(x3)               # 162*256
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        

        x = self.upconv162_642(x)  # 642* 256
        x = torch.cat((x, x3), 1)  # 642 * 512
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
    
    
    
class UNet(nn.Module):

    def __init__(self, num_classes, conv_type, pooling_type):
        super(UNet, self).__init__()
        
        neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42 = Get_indices_order()
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        weight_10242, weight_2562, weight_642, weight_162, weight_42 = Get_weights()
        upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        self.feature_dims = [4, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]

        if conv_type == "RePa":
            conv_layer = gCNN_conv_layer
        if conv_type == "DiNe":
            conv_layer = DiNe_conv_layer
        
        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242, neigh_indices_10242, weight_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats)
        
        self.pool1 = pool_layer(2562, neigh_orders_10242, pooling_type)
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562,  neigh_indices_2562, weight_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562,  neigh_indices_2562, weight_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        
        self.pool2 = pool_layer(642, neigh_orders_2562, pooling_type)
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642,neigh_indices_642, weight_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642,neigh_indices_642, weight_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        
        self.pool3 = pool_layer(162, neigh_orders_642, pooling_type)
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        
        self.pool4 = pool_layer(42, neigh_orders_162, pooling_type)
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[9], neigh_orders_42, neigh_indices_42, weight_42)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[9], self.feature_dims[10], neigh_orders_42, neigh_indices_42, weight_42)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        
        self.upconv42_162 = upconv_layer(162, self.feature_dims[10], self.feature_dims[8], upconv_top_index_162, upconv_down_index_162)
        self.conv6_1 = conv_layer(self.feature_dims[8] * 2, self.feature_dims[8], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_162, neigh_indices_162, weight_162)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)
        
        self.upconv162_642 = upconv_layer(642, self.feature_dims[8], self.feature_dims[6], upconv_top_index_642, upconv_down_index_642)
        self.conv7_1 = conv_layer(self.feature_dims[6] * 2, self.feature_dims[6], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[6], self.feature_dims[6], neigh_orders_642, neigh_indices_642, weight_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.upconv642_2562 = upconv_layer(2562, self.feature_dims[6], self.feature_dims[4], upconv_top_index_2562, upconv_down_index_2562)
        self.conv8_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_orders_2562, neigh_indices_2562, weight_2562)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.upconv2562_10242 = upconv_layer(10242, self.feature_dims[4], self.feature_dims[2], upconv_top_index_10242, upconv_down_index_10242)
        self.conv9_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2], neigh_orders_10242,  neigh_indices_10242, weight_10242)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_orders_10242,  neigh_indices_10242, weight_10242)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[2], num_classes, neigh_orders_10242,   neigh_indices_10242, weight_10242)
        
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
