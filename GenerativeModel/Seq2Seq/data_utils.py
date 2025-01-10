# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:05:45 2024

@author: exy029
"""
import torch
import numpy as np 
from typing import Any, Dict, Collection, List
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
                
def build_vocab(sequences: Collection, max_input_length: int,
                       reverse_input: bool, input_dict: Dict):
    
    """Create the array describing the input sequences.

    These values can be used to generate embeddings for further processing.

    Models used in:

    * SeqToSeq

    Parameters
    ----------
    sequences: Collection
        List of sequences to be converted into input array.
    reverse_input: bool
        If True, reverse the order of input sequences before sending them into
        the encoder. This can improve performance when working with long sequences.
    batch_size: int
        Batch size of the input array.
    input_dict: dict
        Dictionary containing the key-value pairs of input sequences.
    Returns
    -------
    features: np.Array
        Numeric Representation of the given sequence according to input_dict.
    """
    if reverse_input:
        sequences = [reversed(s) for s in sequences]
    features = np.zeros((len(sequences), max_input_length), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for j, token in enumerate(sequence):
            features[i, j] = input_dict[token]
    
    
    return features