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

from model import Unet
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
    parser = argparse.ArgumentParser(description='Predict parcellation map with 36 ROIs based on FreeSurfer protocol from input surfaces',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='trained_models/left_hemi_40k_curv_sulc.pkl',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filename of input surface')
    parser.add_argument('--in_folder', '-in_folder', default='surfaces/left_hemisphere',
                        metavar='INPUT',
                        help='folder path for input files. Will parcelalte all the files end in .vtk in this folder. Accept input or in_folder.')
    parser.add_argument('--output', '-o', metavar='INPUT',
                        help='Filename of ouput surface. If not given, default is [input].parc.vtk')
    parser.add_argument('--out_folder', '-out_folder', metavar='INPUT',
                        help='folder path for ouput surface. If not given, default is the same as input_folder. Accept output or out_folder.')

    args =  parser.parse_args()
    in_file = args.input
    out_file = args.output
    in_folder = args.in_folder
    out_folder = args.out_folder
    model_path = args.model

    if in_file is not None and in_folder is not None:
        raise NotImplementedError('Only need in_put or in_folder. Not both.')
    if in_file is None and in_folder is None:
        raise NotImplementedError('Need in_put or in_folder!')
    if in_file is not None and out_file is None:
        out_file = in_file[0:-4] + '.parc.vtk'
    if in_folder is not None and out_folder is None:
        out_folder = in_folder
    
    model = Unet(2, 36)
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
       
    par_fs_to_36 = get_par_fs_to_36()
    par_36_to_fs = dict(zip(par_fs_to_36.values(), par_fs_to_36.keys()))
    par_36_to_fs_vec = get_par_36_to_fs_vec()

    if in_file is not None:
        data = read_vtk(in_file)
        curv = torch.from_numpy(data['curv'][0:40962]).unsqueeze(1) # use curv data with 40k vertices
        sulc = torch.from_numpy(data['sulc'][0:40962]).unsqueeze(1) # use sulc data with 40k vertices
        pred = inference(curv, sulc, model)
        data['par_fs'] = np.array([par_36_to_fs[i] for i in pred])
        data['par_fs_vec'] = np.array([par_36_to_fs_vec[i] for i in pred])
        write_vtk(data, out_file)
   
    else:
        test_dataset = BrainSphere(in_folder)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        for batch_idx, (data, file) in enumerate(test_dataloader):
            for key, value in data.items():
                data[key] = value.squeeze(0)
            file = file[0]    
            curv = data['curv'][0:40962].unsqueeze(1) # use curv data with 40k vertices
            sulc = data['sulc'][0:40962].unsqueeze(1) # use sulc data with 40k vertices
            for key, value in data.items():
                data[key] = value.numpy()
                
            pred = inference(curv, sulc, model)
            data['par_fs'] = np.array([par_36_to_fs[i] for i in pred])
            data['par_fs_vec'] = np.array([par_36_to_fs_vec[i] for i in pred])

            write_vtk(data, file.replace('.vtk', '.parc.vtk'))

