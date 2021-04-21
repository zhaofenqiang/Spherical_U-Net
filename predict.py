#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import torch
import argparse
import torchvision
import numpy as np
import glob
import os

from model import Unet_40k, Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.utils import get_par_36_to_fs_vec
from sphericalunet.utils.interp_numpy import resampleSphereSurf

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
    feats = feats.to(device)
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
                        help="Specify the level of the surfaces' resolution. Generally, level 7 with 40962 vertices is sufficient, level 8 with 163842 vertices is more accurate but slower.")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filename of input surface')
    parser.add_argument('--output', '-o',  default='[input].parc.vtk', metavar='OUTPUT',
                        help='Filename of ouput surface.')
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='the device for running the model.')

    args =  parser.parse_args()
    in_file = args.input
    out_file = args.output
    hemi = args.hemisphere
    level = args.level   
   
    device = args.device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')

    if in_file is None:
        raise NotImplementedError('Only need in_put filename')
    if in_file is not None and out_file=='[input].parc.vtk':
        out_file = in_file[0:-4] + '.parc.vtk'
    
    if level == '7':
        model = Unet_40k(2, 36)
        model_path = '40k_curv_sulc.pkl'
        n_vertices = 40962
    else:
        model = Unet_160k(2, 36)
        model_path = '160k_curv_sulc.pkl'
        n_vertices = 163842
    
    model_path = 'trained_models/' + hemi + '_hemi_' +  model_path
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
       
    par_36_to_fs_vec = get_par_36_to_fs_vec()

    template = read_vtk('neigh_indices/sphere_' + str(n_vertices) + '.vtk')
    if in_file is not None:
        orig_surf = read_vtk(in_file)
        curv_temp = orig_surf['curv']
        if len(curv_temp) != n_vertices:
            sucu = resampleSphereSurf(orig_surf['vertices'], template['vertices'], 
                                      np.concatenate((orig_surf['sulc'][:,np.newaxis], 
                                                      orig_surf['curv'][:,np.newaxis]),
                                                     axis=1))
            sulc = sucu[:,0]
            curv = sucu[:,1]
        else:
             curv = orig_surf['curv'][0:n_vertices]
             sulc = orig_surf['sulc'][0:n_vertices]
        
        curv = torch.from_numpy(curv).unsqueeze(1) 
        sulc = torch.from_numpy(sulc).unsqueeze(1)
        
        pred = inference(curv, sulc, model)
        pred = par_36_to_fs_vec[pred]
        
        orig_lbl = resample_label(template['vertices'], orig_surf['vertices'], pred)
        
        orig_surf['par_fs_vec'] = orig_lbl
        write_vtk(orig_surf, out_file)
   
