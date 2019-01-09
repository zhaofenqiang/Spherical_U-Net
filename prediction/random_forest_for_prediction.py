#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:33:56 2018

@author: fenqiang
"""

import glob
import os
import numpy as np
import scipy.io as sio 
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


fold1 = '/pine/scr/f/e/fenqiang/prediction/MissingDataPredictionSmoothed150_haar_100/fold1'
fold2 = '/pine/scr/f/e/fenqiang/prediction/MissingDataPredictionSmoothed150_haar_100/fold2'
fold3 = '/pine/scr/f/e/fenqiang/prediction/MissingDataPredictionSmoothed150_haar_100/fold3'
fold4 = '/pine/scr/f/e/fenqiang/prediction/MissingDataPredictionSmoothed150_haar_100/fold4'
fold5 = '/pine/scr/f/e/fenqiang/prediction/MissingDataPredictionSmoothed150_haar_100/fold5'

for f in range(5):
    print("fold: ", f)
    train_fold =[1,2,3,4,5]
    val_fold = f + 1
    train_fold.remove(f+1)
    
    train_files = sorted(glob.glob(os.path.join(locals()['fold' + str(train_fold[0])], '*.Y0')))    
    train_files = train_files + sorted(glob.glob(os.path.join(locals()['fold' + str(train_fold[1])], '*.Y0')))
    train_files = train_files + sorted(glob.glob(os.path.join(locals()['fold' + str(train_fold[2])], '*.Y0')))
    train_files = train_files + sorted(glob.glob(os.path.join(locals()['fold' + str(train_fold[3])], '*.Y0')))
    val_files = sorted(glob.glob(os.path.join(locals()['fold' + str(val_fold)], '*.Y0')))
    
    train_Y0 = np.zeros((len(train_files), 40962, 102))
    train_Y1 = np.zeros((len(train_files), 40962))
    val_Y0 = np.zeros((len(val_files), 40962, 102))
    val_Y1 = np.zeros((len(val_files), 40962))
    
    
    for k in range(len(train_files)):
        file = train_files[k]
        raw_Y0 = sio.loadmat(file)
    #    feats_Y0_sulc = raw_Y0['data'][:,1]
    #    feats_Y0_thick = raw_Y0['data'][:,2]
    #    feats_Y0_sulc = (feats_Y0_sulc - 0.0081) / 0.5169
    #    feats_Y0_thick = (feats_Y0_thick - 1.8586) / 0.4395
    #    train_Y0[i,:] = np.concatenate((feats_Y0_sulc, feats_Y0_thick), 0)
        feats_Y0 = raw_Y0['data'][:,1:]
        train_Y0[k,:,:] = feats_Y0
        
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        feats_Y1 = raw_Y1['data'][:,1]   # 0: curv, 1: sulc, 2: thickness
        train_Y1[k,:] = feats_Y1
        
    for k in range(len(val_files)): 
        file = val_files[k]
        raw_Y0 = sio.loadmat(file)
    #    feats_Y0_sulc = raw_Y0['data'][:,1]
    #    feats_Y0_thick = raw_Y0['data'][:,2]
    #    feats_Y0_sulc = (feats_Y0_sulc - 0.0081) / 0.5169
    #    feats_Y0_thick = (feats_Y0_thick - 1.8586) / 0.4395
    #    val_Y0[i,:] = np.concatenate((feats_Y0_sulc, feats_Y0_thick), 0)
        feats_Y0 = raw_Y0['data'][:,1:]
        val_Y0[k,:,:]  = feats_Y0
        
        raw_Y1 = sio.loadmat(file[:-3] + '.Y1')
        feats_Y1 = raw_Y1['data'][:,1]   # 0: curv, 1: sulc, 2: thickness
        val_Y1[k,:] = feats_Y1
    
    
    # Random forest
    val_thick_Y1_pred = np.zeros((len(val_files),40962))
    regr = RandomForestRegressor(max_depth=70, min_samples_split=3, n_jobs=-1, random_state=0, n_estimators=100)
    for i in range(40962):
        if i % 10 == 0:
            print(i)
        x = train_Y0[:,i,:]
        y = train_Y1[:,i]
        regr.fit(x, y)
        val_thick_Y1_pred[:,i] = regr.predict(val_Y0[:,i,:])
        
    mae = np.absolute(val_Y1 - val_thick_Y1_pred)
    print("RF: mae mean:", np.mean(mae))
    print("RF: mae std:", np.std(mae))
    mre = np.absolute(val_Y1 - val_thick_Y1_pred)/val_Y1
    print("RF: mre mean: ", np.mean(mre))
    print("RF: mre std: ", np.std(mre))    
    
    if val_fold == 1:
        for i in range(len(val_files)):
            file = val_files[i]
            np.savetxt('./pred/RF_pred_' + file.split('/')[-1][0:-3] + '_sulc' + '.txt', val_thick_Y1_pred[i,:])
     


