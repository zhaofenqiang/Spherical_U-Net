#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

@author: fenqiang
"""

import torch
import torch.nn as nn
from torch.nn import init
import argparse
import torchvision
import numpy as np
import glob
import os

from model import Unet_40k, Unet_160k
from vtk_io import read_vtk, write_vtk
from utils import get_par_fs_to_36, get_par_36_to_fs_vec


class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1):

        self.files = sorted(glob.glob(os.path.join(root1, '*.vtk')))    

    def __getitem__(self, index):
        file = self.files[index]
        data = read_vtk(file)
       
        return data, file

    def __len__(self):
        return len(self.files)


def inference(curv, sulc, model):
    feats =torch.cat((curv, sulc), 1)
    feat_max = [1.2, 13.7]
    for i in range(feats.shape[1]):
        feats[:,i] = feats[:, i]/feat_max[i]
    feats = feats.cuda()
    with torch.no_grad():
        prediction = model(feats)
    pred = prediction.max(1)[1]
    pred = pred.cpu().numpy()
    return pred


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Predict the parcellation maps with 36 regions from the input surfaces',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hemisphere', '-hemi', default='left',
                        choices=['left', 'right'], 
                        help="Specify the hemisphere for parcellation, left or right.")
    parser.add_argument('--level', '-l', default='7',
                        choices=['7', '8'],
                        help="Specify the level of the surfaces. Generally, level 7 spherical surface is with 40962 vertices, 8 is with 163842 vertices.")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filename of input surface')
    parser.add_argument('--in_folder', '-in_folder',
                        metavar='INPUT_FOLDER',
                        help='folder path for input files. Will parcelalte all the files end in .vtk in this folder. Accept input or in_folder.')
    parser.add_argument('--output', '-o',  default='[input].parc.vtk', metavar='OUTPUT',
                        help='Filename of ouput surface.')
    parser.add_argument('--out_folder', '-out_folder', default='[in_folder]', metavar='OUT_FOLDER',
                        help='folder path for ouput surface. Accept output or out_folder.')

    args =  parser.parse_args()
    in_file = args.input
    out_file = args.output
    in_folder = args.in_folder
    out_folder = args.out_folder
    hemi = args.hemisphere
    level = args.level   

    if in_file is not None and in_folder is not None:
        raise NotImplementedError('Only need in_put or in_folder. Not both.')
    if in_file is None and in_folder is None:
        raise NotImplementedError('Need in_put or in_folder!')
    if in_file is not None and out_file is None:
        out_file = in_file[0:-4] + '.parc.vtk'
    if in_folder is not None and out_folder is None:
        out_folder = in_folder
    
    if level == '7':
        model = Unet_40k(2, 36)
        model_path = '40k_curv_sulc.pkl'
        n_vertices = 40962
    else:
        model = Unet_160k(2, 36)
        model_path = '160k_curv_sulc.pkl'
        n_vertices = 163842
    
    model_path = 'trained_models/' + hemi + '_hemi_' +  model_path
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
       
    par_fs_to_36 = get_par_fs_to_36()
    par_36_to_fs = dict(zip(par_fs_to_36.values(), par_fs_to_36.keys()))
    par_36_to_fs_vec = get_par_36_to_fs_vec()

    if in_file is not None:
        data = read_vtk(in_file)
        curv_temp = data['curv']
        if len(curv_temp) != n_vertices:
            raise NotImplementedError('Input surfaces level is not consistent with the level '+ level + ' that the model was trained on.')
        curv = torch.from_numpy(data['curv'][0:n_vertices]).unsqueeze(1) # use curv data with 40k vertices
        sulc = torch.from_numpy(data['sulc'][0:n_vertices]).unsqueeze(1) # use sulc data with 40k vertices
        pred = inference(curv, sulc, model)
        data['par_fs'] = np.array([par_36_to_fs[i] for i in pred])
        data['par_fs_vec'] = np.array([par_36_to_fs_vec[i] for i in pred])
        write_vtk(data, out_file)
   
    else:
        test_dataset = BrainSphere(in_folder)
        if len(test_dataset[0][0]['curv']) != n_vertices:
            raise NotImplementedError('Input surfaces level is not consistent with the level '+ level + ' that the model was trained on.')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        for batch_idx, (data, file) in enumerate(test_dataloader):
            for key, value in data.items():
                data[key] = value.squeeze(0)
            file = file[0]    
            curv = data['curv'][0:n_vertices].unsqueeze(1) 
            sulc = data['sulc'][0:n_vertices].unsqueeze(1) 
            for key, value in data.items():
                data[key] = value.numpy()
                
            pred = inference(curv, sulc, model)
            data['par_fs'] = np.array([par_36_to_fs[i] for i in pred])
            data['par_fs_vec'] = np.array([par_36_to_fs_vec[i] for i in pred])

            write_vtk(data, file.replace('.vtk', '.parc.vtk'))

