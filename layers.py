#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:18:30 2018

@author: zfq

This is for brain parcellation. Implement the Spherical U-Net 
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from utils import *


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

    def __init__(self, in_feats, out_feats, neigh_orders, neigh_indices=None, neigh_weights=None):
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
    
class SegNet_pool_layer(nn.Module):

    def __init__(self, num_nodes, neigh_orders, pooling_type):
        super(SegNet_pool_layer, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders
        
    def forward(self, x):
       
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:self.num_nodes*7]].view(self.num_nodes, feat_num, 7)
        x = torch.max(x, 2)
        assert(x[0].size() == torch.Size([self.num_nodes, feat_num]))
                
        return x[0], x[1]


        
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


class upsample_interpolation(nn.Module):

    def __init__(self, num_nodes, upsample_neighs_order):
        super(upsample_interpolation, self).__init__()

        self.num_nodes = num_nodes    
        self.upsample_neighs_order = upsample_neighs_order
       
    def forward(self, x):
       
        feat_num = x.size()[1]
        x1 = x[self.upsample_neighs_order].view(self.num_nodes - x.size()[0], feat_num, 2)
        x1 = torch.mean(x1, 2)
        x = torch.cat((x,x1),0)
                    
        return x

      
class upsample_maxindex(nn.Module):

    def __init__(self, num_nodes, neigh_orders):
        super(upsample_maxindex, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders
        
    def forward(self, x, max_index):
       
        raw_nodes, feat_num = x.size()
        assert(max_index.size() == x.size())
        x = x.view(-1)        
        
        y = torch.zeros(self.num_nodes, feat_num).to(torch.device("cuda"))
        column_ref = torch.zeros(raw_nodes, feat_num)
        for i in range(raw_nodes):
            column_ref[i,:] = i * 7 + max_index[i,:] 
        column_index = self.neigh_orders[column_ref.view(-1).long()]
        column_index = torch.from_numpy(column_index).long()
        row_index = np.floor(np.linspace(0.0, float(feat_num), num=raw_nodes*feat_num))
        row_index[-1] = row_index[-1] - 1
        row_index = torch.from_numpy(row_index).long()
        y[column_index, row_index] = x
        
        return y


 
