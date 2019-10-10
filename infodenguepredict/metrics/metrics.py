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

def compute_metrics(y_true, y_pred):
    """
    Computes metrics scores on predictions and observations.
    
    :param y_true: Array of observations
    :param y_pred: Array of predictions
    :return metrics_scores: Dataframe with scores for chosen metrics 
    """
    metrics_dict = {"explained_variance_score": explained_variance_score,
               "mean_absolute_error": mean_absolute_error,
               "mean_squared_error": mean_squared_error,
               "mean_squared_log_error": mean_squared_log_error,
               "median_absolute_error": median_absolute_error,
               "r2_score": r2_score}
    metrics_scores = {}
    for k,v in metrics_dict.items():
        metrics_scores[k] = v(y_true,y_pred)
    metrics_scores = pd.DataFrame(metrics_scores,index=["scores"],columns=sorted(metrics_scores.keys()))
    return metrics_scores

def compute_residue_predictor_correlations(y_true,y_pred, predictors_df, look_back=1,
                                           predict_n=1):
    """
    Computes correlations between predictors and residues 
    
    :param y_true: Array of observations
    :param y_pred: Array of predictions
    :param predictors_df: Dataframe with predictors
    :param look_back: Number of previous timesteps to use as predictors
    :param predict_n: Number of time steps in the future used to compute predictions. 
    :return metrics_scores: Dataframe with scores for chosen metrics 
    """
    residues = y_true - y_pred
    #corr_scores = pd.DataFrame([],columns = predictors_df.columns)
    corr_scores = []
    for time_back in tqdm(range(1,look_back+1)):
        X = predictors_df[look_back-time_back:-time_back-predict_n+1].T.values
        correlations_at_time_back = []
        #print("shape of x: ", x.shape)
        #print("shape of residues: ", residues.shape)
        for x in X:
            correlations_at_time_back.append(pearsonr(residues,x)[0])
        corr_scores.append(correlations_at_time_back)
    corr_scores = pd.DataFrame(corr_scores,columns=predictors_df.columns,
                               index=["t-{}".format(time_back) for time_back in range(1,look_back+1)])
    return corr_scores    



#def compute_residue_predictor_correlations(y_true,y_pred, X_predictors, predictors, look_back=1,
#                                           predict_n=1,predictor_back=1,as_dataframe=True):
#    """
#    Computes correlations between predictors and residues 
#    """
#    residues = y_true - y_pred
#    corr_scores = {}
#    #for 
#    for p,x in zip(predictors,X_predictors[look_back-predictor_back:-predictor_back-predict_n+1].T):
#        #print("shape of x: ", x.shape)
#        #print("shape of residues: ", residues.shape)
#        corr_scores[p] = pearsonr(residues,x)[0]
#    if as_dataframe:
#        corr_scores = pd.DataFrame(corr_scores,index=["correlations"], columns=sorted(corr_scores.keys()))
#        #corr_scores.index.name = "t+".format(loo)
#    return corr_scores    
        
    
    