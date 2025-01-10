# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:55:36 2024

@author: Rodolfo Freitas
"""

import os
import pandas as pd
import numpy as np
from nn import net_
import deepchem as dc
import torch 
from datetime import datetime
from timeit import default_timer
from math import ceil, floor

from scipy.optimize import minimize
from scipy.optimize import Bounds
from joblib import Parallel, delayed
import multiprocessing

# Generator samples and output normalized prediction
def sample_generator(X, Z): 
    # Evaluate
    net_D.eval()       
    # Prior: p(z)
    z_prior = Z       
    # Decoder: p(y|x,z)
    Y_pred = net_D(torch.cat((X, z_prior),dim=1))      
    return Y_pred

# Generator samples and output de-normalized prediction
def generate_sample(X_star):
    # Fetch latent variables
    Z = torch.randn(X_star.shape[0], 1, device=device)
    
    Y_star = sample_generator(X_star, Z) 
    
    # De-normalize outputs
    Y_star = Y_star 
    
    return Y_star 

def predict(X_star):
    N_samples = 1000
    samples_mean= torch.zeros((X_star.shape[0], N_samples))
    for i in range(0, N_samples): 
        samples_mean[:,i:i+1] = generate_sample(X_star)
    
    # Compute mean and variance of the prediction as function of phi
    mu_pred = torch.mean(samples_mean, dim = 1) * std_CN + mu_CN 
    var_pred = torch.var(samples_mean, dim = 1) * std_CN + mu_CN
    sigma_pred = torch.sqrt(var_pred)
    return mu_pred, sigma_pred


def jac_fun(x, phi, y_true, alpha):
    phi = torch.tensor(phi, dtype=torch.float, requires_grad=True).to(device)
    x = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
    jac = torch.autograd.grad(predict(torch.mm(x.unsqueeze(0),phi)), x)  
    return jac[0].detach().cpu().numpy()
    
def obj_fun(x, phi, y_true, alpha):
    # Mixing Operator
    Phi = torch.tensor(np.matmul(x[None,:], phi), dtype=torch.float).to(device)
    
    # call the surrogate model
    mu, sigma = predict(Phi)
    
    # Lasso
    l1_reg = np.linalg.norm(x, 1)
    loss = np.square(y_true - mu.detach().numpy()) + alpha * l1_reg 
    
    return loss

def opt(k):
    # Constrains x \in [0, 1] & \sum(x) = 1.0
    bounds = Bounds([1]*np.zeros(phi.shape[0]), [1]*np.ones(phi.shape[0]))
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.}
    
    # Initial condition (sampling from a sparse dirichlet distribution) 
    x0 = np.random.dirichlet(np.ones(phi.shape[0]))

    options = {'maxiter': 1000, 'ftol':1e-9}
    sol = minimize(obj_fun, 
                   x0, 
                   method='SLSQP',
                   jac=jac_fun,
                   bounds=bounds,
                   constraints=[cons],
                   options=options,
                   args=(phi,target_fuel,alpha))
    
    x = sol.x 
    objective_funtion = sol.fun
    return x, objective_funtion
    

if __name__ == "__main__":
    date_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    #%% READ MOLECULES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read fuel data [Name, SMILES, CN, METHOD]
    data    = pd.read_excel('/data/fuel_palette.xlsx').to_numpy() 
    fuel_list = data[:14,0] 
    SMILES = data[:14,1].tolist() 
    classes = data[:14,-1].tolist() 

    #%% Mol2Vec Fingerprinters
    '''
    Jaeger, Sabrina, Simone Fulle, and Samo Turk. 
    “Mol2vec: unsupervised machine learning approach with chemical intuition.” 
    Journal of chemical information and modeling 58.1 (2018): 27-35.
    Pre-trained model from https://github.com/samoturk/mol2vec/
    The default model was trained on 20 million compounds downloaded from ZINC 
    using the following paramters.
        - radius 1
        - UNK to replace all identifiers that appear less than 4 times
        - skip-gram and window size of 10
        - embeddings size 300
    '''
    featurizer = dc.feat.Mol2VecFingerprint()
    phi = featurizer.featurize(SMILES)

    #%% Model creation
    # Decoder: p(y|x,z)
    net_D = net_(inp_dim=phi.shape[1] + 1,
                  out_dim=1,
                  n_layers=3,
                  neurons_fc=100,
                  hidden_activation='relu',
                  out_layer_activation=None) # Generator
   
    # load pre-trained models
    load_dir = 'pre-trained'
    net_D.load_state_dict(torch.load(load_dir + "/Decoder_CN.pt"))
    net_D.to(device)

    mu_CN = torch.tensor(torch.load(load_dir + "/data_CN_stats_mean.pt"))
    std_CN = torch.tensor(torch.load(load_dir + "/data_CN_stats_std.pt"))
    
    num_cores = multiprocessing.cpu_count()
    
    # target cetane number fuel
    '''Zhao, Y., Geng, C., E, W. et al. 
     Experimental study on the effects of blending PODEn on performance, 
     combustion and emission characteristics of heavy-duty diesel engines 
     meeting China VI emission standard. Sci Rep 11, 9514 (2021). https://doi.org/10.1038/s41598-021-89057-y
     '''
    target_fuel = 56.5
    fuel = 'Diesel'
    alpha = 1.0 
    K = 1000
    
    # Save diretory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    # Optimization
    output_filename = results_dir + '/output_optimization_fuel_{}_lasso_{}_samples_{}.txt'.format(fuel,alpha,K)
    print(output_filename)

    f = open(output_filename, "w")
    f.write(">>>>>>>>>>>>>>>>>>>>  Output File for Inverse fuel design with scipy minimize  <<<<<<<<<<<<<<<<\n\n")
    f.write(f"Rodolfo Freitas      {date_time}\n")
    f.write(f"GPU name: {torch.cuda.get_device_name(device=device)}\n")
    f.write(f"number of cpus: {num_cores}\n")
    f.write(f"Target fuel CN: {target_fuel}\n")
    f.write("-------------------------------------------------------------------------------------------------------\n\n")
    f.close()
    

    
    time_start = default_timer()
    sol = Parallel(n_jobs=num_cores)(delayed(opt)(k)for k in range(K))
    time_end = default_timer()
    
    mins, secs = divmod(ceil(time_end-time_start), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    
    
    
    f = open(output_filename, "a")
    f.write("-------------------------------------------------------------------------------------------------------\n\n")
    f.write(f"Total Optimization time: {days:02d}-{hours:02d}:{mins:02d}:{secs:02d}\n")
    f.write("-------------------------------------------------------------------------------------------------------\n\n")
    f.close()
    
    
    # save
    torch.save(sol, results_dir+'/compositions_optimization_fuel_{}_lasso_{}_samples_{}.pt'.format(fuel,alpha,K))
    
