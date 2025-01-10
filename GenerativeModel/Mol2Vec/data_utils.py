# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:05:45 2024

@author: exy029
"""
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr, pearsonr

# Fetches a mini-batch of data
def fetch_minibatch(X, Y, batch_size):
    N = Y.shape[0]
    idx = np.random.choice(N, batch_size, replace=False)
    X_batch = X[idx,:]
    # randomly chose the CN from the dataset
    batch_ =  Y[idx,:]
    Y_batch = np.zeros((batch_size,1))
    for idx in range(batch_size):
        mask = batch_[idx] != 'nan'
        Y_batch[idx] = np.random.choice(batch_[idx, mask], 1)
    
    return X_batch, Y_batch



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if np.abs(validation_loss - train_loss) < self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True