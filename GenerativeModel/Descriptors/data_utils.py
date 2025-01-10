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

def remove_correlated_features(x, y, threshold=0.7, method='pearson'):
    '''
    Objective:
        Remove correlated features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed, default=0.7
        method: Correlation method, default: pearson

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Calculate the correlation between descriptors and the physical properties
    corr_ = np.zeros(x.shape[1])
    if method == 'pearson':
        for idx in range(x.shape[1]):
            corr_[idx],_ = pearsonr(x.to_numpy()[:,idx],y)
    elif method == 'spearman':
        for idx in range(x.shape[1]):
            corr_[idx],_ = spearmanr(x.to_numpy()[:,idx],y)
        
    # argsort descending order
    sort_values = abs(corr_).argsort()[::-1][:len(corr_)]

    x = x.reindex(columns=list(x.keys()[sort_values]))
    
    # Calculate the correlation matrix
    corr_matrix = x.corr('spearman')
    
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    
    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if np.abs(val) >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    
    return x



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