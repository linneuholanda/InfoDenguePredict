#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:00:07 2019

@author: rio
"""

import numpy as np
import pandas as pd
import pickle
import math
import os
import shap
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Tesla K40
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from sklearn.metrics import *
from time import time
from tqdm import tqdm
from scipy.stats import pearsonr

def compute_metrics(y_true, y_pred, as_dataframe=True):
    metrics_dict = {"explained_variance_score": explained_variance_score,
               "mean_absolute_error": mean_absolute_error,
               "mean_squared_error": mean_squared_error,
               "mean_squared_log_error": mean_squared_log_error,
               "median_absolute_error": median_absolute_error,
               "r2_score": r2_score}
    metrics_scores = {}
    for k,v in metrics_dict.items():
        metrics_scores[k] = v(y_true,y_pred)
    if as_dataframe:
        metrics_scores = pd.DataFrame(metrics_scores,index=["scores"],columns=sorted(metrics_scores.keys()))
    return metrics_scores

def compute_residue_predictor_correlations_2(y_true,y_pred, predictors_df, look_back=1,
                                           predict_n=1,as_dataframe=True):
    """
    Computes correlations between predictors and residues 
    """
    residues = y_true - y_pred
    #corr_scores = pd.DataFrame([],columns = predictors_df.columns)
    corr_scores = []
    for time_back in tqdm(range(1,look_back+1)):
        x = predictors_df[look_back-time_back:-time_back-predict_n+1].T.values
        correlations_at_time_back = []
        for predictor in predictors_df.columns:
            correlations_at_time_back.append(pearsonr(residues,x)[0])
        corr_scores.append(correlations_at_time_back)
    corr_scores = pd.DataFrame(correlations_at_time_back,columns=predictors_df.columns,
                               index=["t-{}".format(time_back) for time_back in range(1,look_back+1)])
    return corr_scores    



def compute_residue_predictor_correlations(y_true,y_pred, X_predictors, predictors, look_back=1,
                                           predict_n=1,predictor_back=1,as_dataframe=True):
    """
    Computes correlations between predictors and residues 
    """
    residues = y_true - y_pred
    corr_scores = {}
    #for 
    for p,x in zip(predictors,X_predictors[look_back-predictor_back:-predictor_back-predict_n+1].T):
        #print("shape of x: ", x.shape)
        #print("shape of residues: ", residues.shape)
        corr_scores[p] = pearsonr(residues,x)[0]
    if as_dataframe:
        corr_scores = pd.DataFrame(corr_scores,index=["correlations"], columns=sorted(corr_scores.keys()))
        #corr_scores.index.name = "t+".format(loo)
    return corr_scores    
        
    
    