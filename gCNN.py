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


class naive_gCNN(nn.Module):

    def __init__(self, num_classes):
        super(naive_gCNN, self).__init__()
        
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
       
        self.feature_dims = [3, 64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 128, 128, 128, 128]
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

        self.feature_dims = [3, 32, 32, 64, 64, 128, 128, 256, 256]
        
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
        x = self.conv1_1(x)                  # 10242*64
        x = self.relu(self.bn1_1(x))
        x = self.conv1_2(x)                  # 10242*64
        x1 = self.relu(self.bn1_2(x))
        
        x = self.pool1(x1)                  # 2562*64
        x = self.conv2_1(x)                 # 2562*128
        x = self.relu(self.bn2_1(x))
        x = self.conv2_2(x)                 # 2562*128
        x2 = self.relu(self.bn2_2(x))
        
        x = self.pool2(x2)               # 642*128
        x = self.conv3_1(x)               # 642 * 256
        x = self.relu(self.bn3_1(x))
        x = self.conv3_2(x)               # 642 * 256
        x3 = self.relu(self.bn3_2(x))
        
        x = self.pool3(x3)               # 162*256
        x = self.conv4_1(x)                # 162*512
        x = self.relu(self.bn4_1(x))
        x = self.conv4_2(x)
        x = self.relu(self.bn4_2(x))
        

        x = self.upconv162_642(x)  # 642* 256
        x = torch.cat((x, x3), 1)  # 642 * 512
        x = self.conv7_1(x)       # 642 * 256 
        x = self.relu(self.bn7_1(x)) 
        x = self.conv7_2(x)  # 642 * 256 
        x = self.relu(self.bn7_2(x))  # 642 * 256 

        x = self.upconv642_2562(x)  # 2562* 128
        x = torch.cat((x, x2), 1) # 2562 * 256
        x = self.conv8_1(x)   # 2562 * 128 
        x = self.relu(self.bn8_1(x)) 
        x = self.conv8_2(x)  # 2562 * 128 
        x = self.relu(self.bn8_2(x))  # 2562 * 128 
        
        x = self.upconv2562_10242(x)  # 10242* 64
        x = torch.cat((x, x1), 1) # 10242 * 128
        x = self.conv9_1(x)  # 10242 * 64 
        x = self.relu(self.bn9_1(x)) 
        x = self.conv9_2(x)  # 10242 * 64 
        x = self.relu(self.bn9_2(x))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x
    
    
    
    
class UNet(nn.Module):

    def __init__(self, num_classes, conv_layer, pooling_type):
        super(UNet, self).__init__()
        
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index()

        self.feature_dims = [3, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]

        conv_layer = DiNe_conv_layer
        
        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242, None, None)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242, None, None)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats)
        
        self.pool1 = pool_layer(2562, neigh_orders_10242, pooling_type)
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562, None, None)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562, None, None)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        
        self.pool2 = pool_layer(642, neigh_orders_2562, pooling_type)
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642, None, None)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642, None, None)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        
        self.pool3 = pool_layer(162, neigh_orders_642, pooling_type)
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162, None, None)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162, None, None)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        
        self.pool4 = pool_layer(42, neigh_orders_162, pooling_type)
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[9], neigh_orders_42, None, None)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[9], self.feature_dims[10], neigh_orders_42, None, None)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        
        self.upconv42_162 = upconv_layer(162, self.feature_dims[10], self.feature_dims[8], upconv_top_index_162, upconv_down_index_162)
        self.conv6_1 = conv_layer(self.feature_dims[8] * 2, self.feature_dims[8], neigh_orders_162, None, None)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_162, None, None)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)
        
        self.upconv162_642 = upconv_layer(642, self.feature_dims[8], self.feature_dims[6], upconv_top_index_642, upconv_down_index_642)
        self.conv7_1 = conv_layer(self.feature_dims[6] * 2, self.feature_dims[6], neigh_orders_642, None, None)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[6], self.feature_dims[6], neigh_orders_642, None, None)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.upconv642_2562 = upconv_layer(2562, self.feature_dims[6], self.feature_dims[4], upconv_top_index_2562, upconv_down_index_2562)
        self.conv8_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4], neigh_orders_2562, None, None)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_orders_2562, None, None)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.upconv2562_10242 = upconv_layer(10242, self.feature_dims[4], self.feature_dims[2], upconv_top_index_10242, upconv_down_index_10242)
        self.conv9_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2], neigh_orders_10242, None, None)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_orders_10242, None, None)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[2], num_classes, neigh_orders_10242, None, None)
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(self.bn1_1(x))
        x = self.conv1_2(x)
        x1 = self.relu(self.bn1_2(x))
        
        x = self.pool1(x1)
        x = self.conv2_1(x)
        x = self.relu(self.bn2_1(x))
        x = self.conv2_2(x)
        x2 = self.relu(self.bn2_2(x))
        
        x = self.pool2(x2)          
        x = self.conv3_1(x)
        x = self.relu(self.bn3_1(x))
        x = self.conv3_2(x)
        x3 = self.relu(self.bn3_2(x))
        
        x = self.pool3(x3)
        x = self.conv4_1(x)
        x = self.relu(self.bn4_1(x))
        x = self.conv4_2(x)
        x4 = self.relu(self.bn4_2(x))
        
        x = self.pool4(x4)
        x = self.conv5_1(x)
        x = self.relu(self.bn5_1(x))
        x = self.conv5_2(x)
        x = self.relu(self.bn5_2(x))
        
        x = self.upconv42_162(x)  # 162* 512
        x = torch.cat((x, x4), 1) # 162 * 1024
        x = self.conv6_1(x)  # 162 * 512 
        x = self.relu(self.bn6_1(x)) 
        x = self.conv6_2(x)  # 162 * 512 
        x = self.relu(self.bn6_2(x))  # 162 * 512

        x = self.upconv162_642(x)  # 642* 256
        x = torch.cat((x, x3), 1) # 642 * 512
        x = self.conv7_1(x)  # 642 * 256 
        x = self.relu(self.bn7_1(x)) 
        x = self.conv7_2(x)  # 642 * 256 
        x = self.relu(self.bn7_2(x))  # 642 * 256 

        x = self.upconv642_2562(x)  # 2562* 128
        x = torch.cat((x, x2), 1) # 2562 * 256
        x = self.conv8_1(x)  # 2562 * 128 
        x = self.relu(self.bn8_1(x)) 
        x = self.conv8_2(x)  # 2562 * 128 
        x = self.relu(self.bn8_2(x))  # 2562 * 128 
        
        x = self.upconv2562_10242(x)  # 10242* 64
        x = torch.cat((x, x1), 1) # 10242 * 128
        x = self.conv9_1(x)  # 10242 * 64 
        x = self.relu(self.bn9_1(x)) 
        x = self.conv9_2(x)  # 10242 * 64 
        x = self.relu(self.bn9_2(x))  # 10242 * 64 
        
        x = self.conv10(x) # 10242 * 36
        
        return x
    
class gCNN_conv_layer(nn.Module):

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices, neigh_weights):
        super(gCNN_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_indices = neigh_indices.reshape(-1) - 1
        self.weight = nn.Linear(25 * in_feats, out_feats)
        self.nodes_number = neigh_indices.shape[0]
        
        self.neigh_weights = np.reshape(np.tile(neigh_weights, self.in_feats), (neigh_weights.shape[0],neigh_weights.shape[1],3,-1)).astype(np.float32)
        self.neigh_weights = torch.from_numpy(self.neigh_weights).cuda()    
        
    def forward(self, x):
      
        mat = x[self.neigh_indices]
        mat = mat.view(self.nodes_number, 25, 3, -1)
        assert(mat.size() == torch.Size([self.nodes_number, 25, 3, self.in_feats]))
   
        assert(mat.size() == self.neigh_weights.size())

        x = torch.mul(mat, self.neigh_weights)
        x = torch.sum(x, 2).view(self.nodes_number, -1)
        assert(x.size() == torch.Size([self.nodes_number, 25 * self.in_feats]))
        
        out = self.weight(x)
        return out


class DiNe_conv_layer(nn.Module):

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices, neigh_weights):
        super( DiNe_conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(7 * in_feats, out_feats)
        
    def forward(self, x):
       
        mat = x[self.neigh_orders].view(len(x), 7*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features
    

class pool_layer(nn.Module):

    def __init__(self, num_nodes, neigh_orders, pooling_type):
        super(pool_layer, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders
        self.pooling_type = pooling_type
        
    def forward(self, x):
       
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:self.num_nodes*7]].view(self.num_nodes, feat_num, 7)
        if self.pooling_type == "mean":
            x = torch.mean(x, 2)
        if self.pooling_type == "max":
            x = torch.max(x, 2)[0]
        assert(x.size() == torch.Size([self.num_nodes, feat_num]))
                
        return x
    
        
class upconv_layer(nn.Module):

    def __init__(self, num_nodes, in_feats, out_feats, upconv_top_index, upconv_down_index):
        super(upconv_layer, self).__init__()

        self.num_nodes = num_nodes
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.upconv_top_index = upconv_top_index
        self.upconv_down_index = upconv_down_index
        self.weight = nn.Linear(in_feats, 7 * out_feats)
        
    def forward(self, x):
       
        raw_nodes = x.size()[0]
        x = self.weight(x)
        x = x.view(len(x) * 7, self.out_feats)
        x1 = x[self.upconv_top_index]
        assert(x1.size() == torch.Size([raw_nodes, self.out_feats]))
        x2 = x[self.upconv_down_index].view(-1, self.out_feats, 2)
        x = torch.cat((x1,torch.mean(x2, 2)), 0)
        assert(x.size() == torch.Size([self.num_nodes, self.out_feats]))
        return x





 
