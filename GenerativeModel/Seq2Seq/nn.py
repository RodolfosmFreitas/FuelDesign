#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:18:02 2023

@author: rodolfofreitas

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class net_(nn.Module):
    """
    An implementation of a Fully-Connected Neural Network (Pytorch)
    """
    def __init__(self, inp_dim=5, 
                 out_dim=1,
                 n_layers=3, 
                 neurons_fc = 100,
                 hidden_activation='tanh',
                 out_layer_activation=None):
        
        super(net_, self).__init__()
        self.n_layers = n_layers
        self.input_dim = inp_dim
        self.n_layers = n_layers
        self.output_dim = out_dim
        self.neurons = neurons_fc
        
        # Define the output activation function
        if out_layer_activation is None:
            self.final_layer_activation = nn.Identity()
        elif out_layer_activation == 'sigmoid':
            self.final_layer_activation = nn.Sigmoid()
        elif out_layer_activation == 'softplus':
            self.final_layer_activation = nn.Softplus()
            
        # Define activation function hidden layer
        if hidden_activation is None:
            self.activation = nn.Identity()
        elif hidden_activation == 'relu':
            self.activation = nn.ReLU()
        elif hidden_activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif hidden_activation == 'elu':
            self.activation = nn.ELU()
        elif hidden_activation == 'gelu':
            self.activation = nn.GELU()
        elif hidden_activation == 'tanh':
            self.activation = nn.Tanh()
        
        # define FC layers
        self.fc_net = nn.Sequential()
        # Input layer
        self.fc_net.add_module('fc_inp', nn.Linear(self.input_dim, self.neurons))
        self.fc_net.add_module('activation', self.activation)
        # Hidden Layers
        for i in range(self.n_layers):
            layername = 'hidden_layer{}'.format(i+1)
            self.fc_net.add_module(layername, nn.Linear(self.neurons, self.neurons))
            self.fc_net.add_module('activation{}'.format(i+1), self.activation)
            
        # output layer
        self.fc_net.add_module('fc_out', nn.Linear(self.neurons, self.output_dim))
        self.fc_net.add_module('output_activation', self.final_layer_activation)
        
        # Initialize the Net
        self.initialize_weights()
        
    # Initialize network weights and biases using Xavier initialization
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    
    def forward(self, x):
        """
        Implement the forward pass of the NN. 
        """
        x = self.fc_net(x)
        
        return x
    
    # count the number of parameters
    def num_parameters(self):
        n_params, n_hidden_layers = 0, 0 
        for name, param in self.named_parameters():
            if 'hidden_layer' in name:
                n_hidden_layers +=1
            n_params += param.numel()
        return n_params, n_hidden_layers//2 # It counts bias as a layer, so divide by 2 
    
    


