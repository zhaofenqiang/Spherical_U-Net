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
from bscnn.utils import Get_neighs_order, Get_upsample_neighs_order, Get_upconv_index


class BrainSphereCNN(nn.Module):

    def __init__(self, num_classes):
        super(BrainSphereCNN, self).__init__()
        
        neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
        upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = Get_upconv_index() 

        self.feature_dims = [3, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
        
        self.relu = nn.ReLU()
        self.conv1_1 = conv_layer(self.feature_dims[0], self.feature_dims[1], neigh_orders_10242)
        self.bn1_1 = nn.BatchNorm1d(self.conv1_1.out_feats)
        self.conv1_2 = conv_layer(self.feature_dims[1], self.feature_dims[2], neigh_orders_10242)
        self.bn1_2 = nn.BatchNorm1d(self.conv1_2.out_feats)
        
        self.pool1 = pool_layer(2562, neigh_orders_10242)
        self.conv2_1 = conv_layer(self.feature_dims[2], self.feature_dims[3], neigh_orders_2562)
        self.bn2_1 = nn.BatchNorm1d(self.conv2_1.out_feats)
        self.conv2_2 = conv_layer(self.feature_dims[3], self.feature_dims[4], neigh_orders_2562)
        self.bn2_2 = nn.BatchNorm1d(self.conv2_2.out_feats)
        
        self.pool2 = pool_layer(642, neigh_orders_2562)
        self.conv3_1 = conv_layer(self.feature_dims[4], self.feature_dims[5], neigh_orders_642)
        self.bn3_1 = nn.BatchNorm1d(self.conv3_1.out_feats)
        self.conv3_2 = conv_layer(self.feature_dims[5], self.feature_dims[6], neigh_orders_642)
        self.bn3_2 = nn.BatchNorm1d(self.conv3_2.out_feats)
        
        self.pool3 = pool_layer(162, neigh_orders_642)
        self.conv4_1 = conv_layer(self.feature_dims[6], self.feature_dims[7], neigh_orders_162)
        self.bn4_1 = nn.BatchNorm1d(self.conv4_1.out_feats)
        self.conv4_2 = conv_layer(self.feature_dims[7], self.feature_dims[8], neigh_orders_162)
        self.bn4_2 = nn.BatchNorm1d(self.conv4_2.out_feats)
        
        self.pool4 = pool_layer(42, neigh_orders_162)
        self.conv5_1 = conv_layer(self.feature_dims[8], self.feature_dims[9], neigh_orders_42)
        self.bn5_1 = nn.BatchNorm1d(self.conv5_1.out_feats)
        self.conv5_2 = conv_layer(self.feature_dims[9], self.feature_dims[10], neigh_orders_42)
        self.bn5_2 = nn.BatchNorm1d(self.conv5_2.out_feats)
        
        
        self.upconv42_162 = upconv_layer(162, self.feature_dims[10], self.feature_dims[8], upconv_top_index_162, upconv_down_index_162)
        self.conv6_1 = conv_layer(self.feature_dims[8] * 2, self.feature_dims[8], neigh_orders_162)
        self.bn6_1 = nn.BatchNorm1d(self.conv6_1.out_feats)
        self.conv6_2 = conv_layer(self.feature_dims[8], self.feature_dims[8], neigh_orders_162)
        self.bn6_2 = nn.BatchNorm1d(self.conv6_2.out_feats)
        
        self.upconv162_642 = upconv_layer(642, self.feature_dims[8], self.feature_dims[6], upconv_top_index_642, upconv_down_index_642)
        self.conv7_1 = conv_layer(self.feature_dims[6] * 2, self.feature_dims[6], neigh_orders_642)
        self.bn7_1 = nn.BatchNorm1d(self.conv7_1.out_feats)
        self.conv7_2 = conv_layer(self.feature_dims[6], self.feature_dims[6], neigh_orders_642)
        self.bn7_2 = nn.BatchNorm1d(self.conv7_2.out_feats)
        
        self.upconv642_2562 = upconv_layer(2562, self.feature_dims[6], self.feature_dims[4], upconv_top_index_2562, upconv_down_index_2562)
        self.conv8_1 = conv_layer(self.feature_dims[4] * 2, self.feature_dims[4], neigh_orders_2562)
        self.bn8_1 = nn.BatchNorm1d(self.conv8_1.out_feats)
        self.conv8_2 = conv_layer(self.feature_dims[4], self.feature_dims[4], neigh_orders_2562)
        self.bn8_2 = nn.BatchNorm1d(self.conv8_2.out_feats)
        
        self.upconv2562_10242 = upconv_layer(10242, self.feature_dims[4], self.feature_dims[2], upconv_top_index_10242, upconv_down_index_10242)
        self.conv9_1 = conv_layer(self.feature_dims[2] * 2, self.feature_dims[2], neigh_orders_10242)
        self.bn9_1 = nn.BatchNorm1d(self.conv9_1.out_feats)
        self.conv9_2 = conv_layer(self.feature_dims[2], self.feature_dims[2], neigh_orders_10242)
        self.bn9_2 = nn.BatchNorm1d(self.conv9_2.out_feats)
        
        self.conv10 = conv_layer(self.feature_dims[2], num_classes, neigh_orders_10242)
        
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
    
class conv_layer(nn.Module):

    def __init__(self, in_feats, out_feats, neigh_orders):
        super(conv_layer, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.neigh_orders = neigh_orders
        
        self.weight = nn.Linear(7 * in_feats, out_feats)
        
    def forward(self, x):
       
        mat = x[self.neigh_orders].view(len(x), 7*self.in_feats)
                
        out_features = self.weight(mat)
        return out_features


class pool_layer(nn.Module):

    def __init__(self, num_nodes, neigh_orders):
        super(pool_layer, self).__init__()

        self.num_nodes = num_nodes
        self.neigh_orders = neigh_orders
        
    def forward(self, x):
       
        feat_num = x.size()[1]
        x = x[self.neigh_orders[0:self.num_nodes*7]].view(self.num_nodes, feat_num, 7)
        x = torch.mean(x, 2)
        #x = torch.max(x, 2)[0]
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






#class upsample_layer(nn.Module):
#
#    def __init__(self, num_nodes, upsample_neighs_order, upsampling_type):
#        super(upsample_layer, self).__init__()
#
#        self.num_nodes = num_nodes
#        self.upsample_neighs_order = upsample_neighs_order
#        self.upsampling_type = upsampling_type
#        
#    def forward(self, x):
#       
#        feat_num = x.size()[1]
#        if self.upsampling_type == 'interpolation':
#            x1 = x[self.upsample_neighs_order].view(self.num_nodes - x.size()[0], feat_num, 2)
#            x1 = torch.mean(x1, 2)
#            x = torch.cat((x,x1),0)
#                    
#            return x
#        
        
        
#%%
#num_classes = 36
#neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = Get_neighs_order()
#upsample_neighs_10242, upsample_neighs_2562, upsample_neighs_642, upsample_neighs_162, upsample_neighs_42 = Get_upsample_neighs_order()
#
#feature_dims = [3, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
#
#relu = nn.ReLU()
#conv1_1 = conv_layer(feature_dims[0], feature_dims[1], neigh_orders_10242)
#bn1_1 = nn.BatchNorm1d(conv1_1.out_feats)
#conv1_2 = conv_layer(feature_dims[1], feature_dims[2], neigh_orders_10242)
#bn1_2 = nn.BatchNorm1d(conv1_2.out_feats)
#
#pool1 = pool_layer(2562, neigh_orders_10242)
#pred_conv1 = conv_layer(feature_dims[2], num_classes, neigh_orders_2562)
#  
#conv2_1 = conv_layer(feature_dims[2], feature_dims[3], neigh_orders_2562)
#bn2_1 = nn.BatchNorm1d(conv2_1.out_feats)
#conv2_2 = conv_layer(feature_dims[3], feature_dims[4], neigh_orders_2562)
#bn2_2 = nn.BatchNorm1d(conv2_2.out_feats)
#
#pool2 = pool_layer(642, neigh_orders_2562)
#pred_conv2 = conv_layer(feature_dims[4], num_classes, neigh_orders_642)
#
#conv3_1 = conv_layer(feature_dims[4], feature_dims[5], neigh_orders_642)
#bn3_1 = nn.BatchNorm1d(conv3_1.out_feats)
#conv3_2 = conv_layer(feature_dims[5], feature_dims[6], neigh_orders_642)
#bn3_2 = nn.BatchNorm1d(conv3_2.out_feats)
#
#pool3 = pool_layer(162, neigh_orders_642)
#pred_conv3 = conv_layer(feature_dims[6], num_classes, neigh_orders_162)
#
#conv4_1 = conv_layer(feature_dims[6], feature_dims[7], neigh_orders_162)
#bn4_1 = nn.BatchNorm1d(conv4_1.out_feats)
#conv4_2 = conv_layer(feature_dims[7], feature_dims[8], neigh_orders_162)
#bn4_2 = nn.BatchNorm1d(conv4_2.out_feats)
#
#pool4 = pool_layer(42, neigh_orders_162)
#pred_conv4 = conv_layer(feature_dims[8], num_classes, neigh_orders_42)
#
#conv5_1 = conv_layer(feature_dims[8], feature_dims[9], neigh_orders_42)
#bn5_1 = nn.BatchNorm1d(conv5_1.out_feats)
#conv5_2 = conv_layer(feature_dims[9], feature_dims[10], neigh_orders_42)
#bn5_2 = nn.BatchNorm1d(conv5_2.out_feats)
#
#pool5 = pool_layer(12, neigh_orders_42)
#pred_conv5 = conv_layer(feature_dims[10], num_classes, neigh_orders_12)
#upsample12_42 = upsample_layer(42, upsample_neighs_42, 'interpolation')
#upsample42_162 = upsample_layer(162, upsample_neighs_162, 'interpolation')
#upsample162_642 = upsample_layer(642, upsample_neighs_642, 'interpolation')
#upsample642_2562 = upsample_layer(2562, upsample_neighs_2562, 'interpolation')
#upsample2562_10242 = upsample_layer(10242, upsample_neighs_10242, 'interpolation')
#
#x.size()
#
#x = conv1_1(x)
#x = relu(bn1_1(x))
#x = conv1_2(x)
#x = relu(bn1_2(x))
#
#x1 = pool1(x)
#pred1 = pred_conv1(x1)
#
#x = conv2_1(x1)
#x = relu(bn2_1(x))
#x = conv2_2(x)
#x = relu(bn2_2(x))
#
#x2 = pool2(x)
#pred2 = pred_conv2(x2)
#  
#x = conv3_1(x2)
#x = relu(bn3_1(x))
#x = conv3_2(x)
#x = relu(bn3_2(x))
#
#x3 = pool3(x)
#pred3 = pred_conv3(x3)
#
#x = conv4_1(x3)
#x = relu(bn4_1(x))
#x = conv4_2(x)
#x = relu(bn4_2(x))
#
#x4 = pool4(x)
#pred4 = pred_conv4(x4)
# 
#x = conv5_1(x4)
#x = relu(bn5_1(x))
#x = conv5_2(x)
#x = relu(bn5_2(x))
#
#x5 = pool5(x)
#pred5 = pred_conv5(x5)
#
#pred5 = upsample12_42(pred5)
#pred4 = pred5 + pred4
#pred4 = upsample42_162(pred4)
#pred3 = pred4 + pred3
#pred3 = upsample162_642(pred3)
#pred2 = pred3 + pred2
#pred2 = upsample642_2562(pred2)
#pred1 = pred1 + pred2
#pred1 = upsample2562_10242(pred1)        
