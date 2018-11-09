#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: zfq
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from utils import Get_neighs_order
from layers import DiNe_conv_layer, upsample_maxindex, SegNet_pool_layer

class SegNet(nn.Module):

    def __init__(self, num_classes, conv_layer, pooling_type):
        super(SegNet, self).__init__()
        
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()

        self.feature_dims = [3, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]

        conv_layer = DiNe_conv_layer
        unpool_layer = upsample_maxindex
        pooling_type = "max"
        pool_layer = SegNet_pool_layer

        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats) 
        self.pool1 = pool_layer(2562, neigh_orders_10242, pooling_type)
        
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        self.pool2 = pool_layer(642, neigh_orders_2562, pooling_type)
        
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        self.conv3_3 = conv_layer(self.feature_dims[6], self.feature_dims[6], neigh_orders_642)
        self.bn3_3 = nn.BatchNorm1d(self.conv3_3.out_feats)
        self.pool3 = pool_layer(162, neigh_orders_642, pooling_type)
        
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        self.conv4_3 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_162)
        self.bn4_3 = nn.BatchNorm1d(self.conv4_3.out_feats)
        self.pool4 = pool_layer(42, neigh_orders_162, pooling_type)
        
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_42)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_42)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        self.unpool1 = unpool_layer(162, neigh_orders_162)
        self.conv6_1 = conv_layer(self.feature_dims[8], self.feature_dims[7], neigh_orders_162)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[7], self.feature_dims[6], neigh_orders_162)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)
        self.conv6_3 = conv_layer(self.feature_dims[7], self.feature_dims[6], neigh_orders_162)
        self.bn6_3 = nn.BatchNorm1d(self.conv6_3.out_feats)
       
        self.unpool2 = unpool_layer(642, neigh_orders_642) 
        self.conv7_1 = conv_layer(self.feature_dims[6], self.feature_dims[5], neigh_orders_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[5], self.feature_dims[4], neigh_orders_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        self.conv7_3 = conv_layer(self.feature_dims[5], self.feature_dims[4], neigh_orders_642)
        self.bn7_3 = nn.BatchNorm1d(self.conv7_3.out_feats)
        
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
        
        self.conv10 = conv_layer(self.feature_dims[1], num_classes, neigh_orders_10242)
        
    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x))) # 10242 * 64
        x = self.relu(self.bn1_2(self.conv1_2(x))) # 10242 * 64
        x, max_index1 = self.pool1(x) # 2562 * 64
        
        x = self.relu(self.bn2_1(self.conv2_1(x))) # 2562 * 128
        x = self.relu(self.bn2_2(self.conv2_2(x))) # 2562 * 128
        x, max_index2 = self.pool2(x)  # 642 * 128
        
        x = self.relu(self.bn3_1(self.conv3_1(x))) # 642 * 256
        x = self.relu(self.bn3_2(self.conv3_2(x))) #  642 * 256
       # x = self.relu(self.bn3_3(self.conv3_3(x))) #  642 * 256
        x, max_index3 = self.pool3(x) # 162 *256
        
        x = self.relu(self.bn4_1(self.conv4_1(x))) # 162 * 512
        x = self.relu(self.bn4_2(self.conv4_2(x))) # 162 * 512
        #x = self.relu(self.bn4_3(self.conv4_3(x))) # 162 * 512
        x, max_index4 = self.pool4(x) # 42 * 512
        
        x = self.relu(self.bn5_1(self.conv5_1(x))) # 42 * 512
        x = self.relu(self.bn5_2(self.conv5_2(x))) # 42 * 512
     
        x = self.unpool1(x, max_index4)  # 162* 512
        x = self.relu(self.bn6_1(self.conv6_1(x))) 
        x = self.relu(self.bn6_2(self.conv6_2(x)))  # 162 * 512
       # x = self.relu(self.bn6_3(self.conv6_3(x)))  # 162 * 256

        x = self.unpool2(x, max_index3)  # 642 * 512
        x = self.relu(self.bn7_1(self.conv7_1(x))) 
        x = self.relu(self.bn7_2(self.conv7_2(x)))  # 642 * 256 
       # x = self.relu(self.bn7_3(self.conv7_3(x)))  # 642 * 256 

        x = self.unpool3(x, max_index2)  # 2562 * 128
        x = self.relu(self.bn8_1(self.conv8_1(x))) 
        x = self.relu(self.bn8_2(self.conv8_2(x)))  # 2562 * 64 
        
        x = self.unpool4(x, max_index1)  # 10242 * 64
        x = self.relu(self.bn9_1(self.conv9_1(x))) 
        x = self.relu(self.bn9_2(self.conv9_2(x)))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x

