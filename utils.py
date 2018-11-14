#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:55:33 2018

@author: zfq
"""

import scipy.io as sio 
import numpy as np
import glob
import os
from numpy import median

def Get_indices_order():
    neigh_indices_10242 = get_indices_order('neigh_indices/rec_neigh_indices_10242.mat')
    neigh_indices_2562 = get_indices_order('neigh_indices/rec_neigh_indices_2562.mat')
    neigh_indices_642 = get_indices_order('neigh_indices/rec_neigh_indices_642.mat')
    neigh_indices_162 = get_indices_order('neigh_indices/rec_neigh_indices_162.mat')
    neigh_indices_42 = get_indices_order('neigh_indices/rec_neigh_indices_42.mat')
    
    return neigh_indices_10242, neigh_indices_2562, neigh_indices_642, neigh_indices_162, neigh_indices_42
  

def get_indices_order(indices_path):
    
    indices = sio.loadmat(indices_path)
    indices = indices[indices_path.split('/')[-1][10:-4]].astype(np.int64)
        
    return indices




def Get_weights():
    weight_10242 = get_weights('neigh_indices/weight_10242.mat')
    weight_2562 = get_weights('neigh_indices/weight_2562.mat')
    weight_642 = get_weights('neigh_indices/weight_642.mat')
    weight_162 = get_weights('neigh_indices/weight_162.mat')
    weight_42 = get_weights('neigh_indices/weight_42.mat')
    
    return weight_10242, weight_2562, weight_642, weight_162, weight_42

def get_weights(weight_path):
    
    weight = sio.loadmat(weight_path)
    weight = weight[weight_path.split('/')[-1][0:-4]]
        
    return weight



def Get_neighs_order():
    neigh_orders_40962 = get_neighs_order('neigh_indices/adj_mat_order_40962.mat')
    neigh_orders_10242 = get_neighs_order('neigh_indices/adj_mat_order_10242.mat')
    neigh_orders_2562 = get_neighs_order('neigh_indices/adj_mat_order_2562.mat')
    neigh_orders_642 = get_neighs_order('neigh_indices/adj_mat_order_642.mat')
    neigh_orders_162 = get_neighs_order('neigh_indices/adj_mat_order_162.mat')
    neigh_orders_42 = get_neighs_order('neigh_indices/adj_mat_order_42.mat')
    neigh_orders_12 = get_neighs_order('neigh_indices/adj_mat_order_12.mat')
    
    return neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
    
def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders



def Get_upconv_index():
    
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index('neigh_indices/adj_mat_order_40962.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index('neigh_indices/adj_mat_order_10242.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index('neigh_indices/adj_mat_order_2562.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index('neigh_indices/adj_mat_order_642.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index('neigh_indices/adj_mat_order_162.mat')
    
    return  upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 


def get_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index


def compute_weight():
    folder = 'neigh_indices/90/raw'
    files = sorted(glob.glob(os.path.join(folder, '*.label')))
    
    labels = np.zeros((len(files),10242))
    for i in range(len(files)):
        file = files[i]
        label = sio.loadmat(file)
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        label = label.astype(np.float64)
        labels[i,:] = label
        
    num = np.zeros(36)
    for i in range(36):
        num[i] = len(np.where(labels == i)[0])
       
    num = num/sum(num) 
    num = median(num)/num
    print(num)

    return num
    


def Get_upsample_neighs_order():
    
    upsample_neighs_10242 = get_upsample_order('neigh_indices/adj_mat.mat',
                                               'neigh_indices/adj_order.mat')
    upsample_neighs_2562 = get_upsample_order('neigh_indices/adj_mat_2562.mat',
                                               'neigh_indices/adj_order_2562.mat')
    upsample_neighs_642 = get_upsample_order('neigh_indices/adj_mat_642.mat',
                                               'neigh_indices/adj_order_642.mat')
    upsample_neighs_162 = get_upsample_order('neigh_indices/adj_mat_162.mat',
                                               'neigh_indices/adj_order_162.mat')
    upsample_neighs_42 = get_upsample_order('neigh_indices/adj_mat_42.mat',
                                               'neigh_indices/adj_order_42.mat')
    
    return upsample_neighs_10242, upsample_neighs_2562, upsample_neighs_642, upsample_neighs_162, upsample_neighs_42
    

def get_upsample_order(mat_path, order_path):
    adj_mat = sio.loadmat(mat_path)
    adj_order = sio.loadmat(order_path)
    adj_mat = adj_mat[mat_path.split('/')[-1][0:-4]]
    adj_order = adj_order[order_path.split('/')[-1][0:-4]]
    nodes = len(adj_mat)
    next_nodes = int((len(adj_mat)+6)/4)
    upsample_neighs_order = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = list(np.nonzero(adj_mat[i]))[0]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        upsample_neighs_order[(i-next_nodes)*2:(i-next_nodes)*2+2]= parent_nodes      
    
    return upsample_neighs_order  
