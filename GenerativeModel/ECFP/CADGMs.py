# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:04:14 2024

@author: exy029
"""

import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io
import time
import json
from scipy import stats
from data_utils import fetch_minibatch, EarlyStopping
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from nn import net_
import deepchem as dc


# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Train
parser = argparse.ArgumentParser(description='CADGMs + Fuel2Fuel Fingerprinters')
parser.add_argument('--data-dir', type=str, default="../../data", help='data directory')
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
parser.add_argument('--train-size', type=float, default=0.85, help='amount of the data used to train')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learnign rate')
parser.add_argument('--n-iterations', type=int, default=30000, help='number of iterations to train (default: 20000)')
parser.add_argument('--log-interval', type=int, default=100, help='how many epochs to wait before logging training status')

# CADGMs
parser.add_argument('--num-layers-AE', type=int, default=3, help='number of FC layers AE')
parser.add_argument('--neurons-AE', type=int, default=100, help='number of neurons in the AE')
parser.add_argument('--num-layers-T', type=int, default=2, help='number of FC layers Discriminator')
parser.add_argument('--neurons-T', type=int, default=100, help='number of neurons in the Discriminator')
parser.add_argument('--features', type=int, default=1, help='latent variable dimension')
parser.add_argument('--activation', type=str, default='relu', help='Hidden layer activation, [relu, elu, gelu, tanh, None=linear]')
parser.add_argument('--output-activation', type=str, default=None, help='Output layer activation, sigmoid ~ [0,1], None=linear, softplus ~ [0, inf[')
parser.add_argument('--lam', type=float, default=1.5, help='Entropic regularization parameter ( > 1)')
parser.add_argument('--beta', type=float, default=0.5, help='regularization parameter')
# Ratio of training generator and discriminator in each iteration: k1 for discriminator, k2 for generator
parser.add_argument('--k1', type=int, default=1, help='Discriminator')
parser.add_argument('--k2', type=int, default=1, help='Generator')

args = parser.parse_args()

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
print("Torch device:{}".format(device))
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

#%% Preparing the data for model

# read fuel data [Name, SMILES, CN, METHOD]
data    = pd.read_excel('{}/data_CoOptima_Compedium.xlsx'.format(args.data_dir)).to_numpy() 
fuel_list = data[:,0]
SMILES = data[:,1].tolist()
# Create a new SMILES list without NaN values
SMILES = [c for c in SMILES if str(c) != 'nan']
fuel_list = [f for f in fuel_list if str(f) != 'nan']

# load the cetane numbers dataset
CN = np.load('{}/cetane_numers_CoOptima_Compedium.npy'.format(args.data_dir))

# ECFP
featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)
feat = featurizer.featurize(SMILES)

# Split the data in train and test dataset
X_train, X_test, Y_train, Y_test, fuel_train, fuel_test = train_test_split(feat,
                                                                           CN,
                                                                           fuel_list,
                                                                           train_size=args.train_size, 
                                                                           shuffle=True,
                                                                           random_state=42)

# pre-processing the data (Scaling)
Y_train = Y_train
Y_test = Y_test
aux_nan = Y_train[~np.isnan(Y_train)]
mu_CN = aux_nan.mean(0)
std_CN = aux_nan.std(0)
Y_train = ((Y_train - mu_CN)/ std_CN).astype(str) 
Y_test = ((Y_test - mu_CN)/ std_CN).astype(str)

#%% MODEL
Y_dim = 1 # cetane number
# Model creation
# Decoder: p(y|x,z)
net_D = net_(inp_dim=X_train.shape[1] + args.features,
             out_dim=Y_dim,
             n_layers=args.num_layers_AE,
             neurons_fc=args.neurons_AE,
             hidden_activation=args.activation,
             out_layer_activation=None) # Generator
print(net_D)
print("Generator: number of parameters {} of layers {}".format(*net_D.num_parameters()))

# Encoder: q(z|x,y)
net_E = net_(inp_dim=X_train.shape[1] + Y_dim,
             out_dim=args.features,
             n_layers=args.num_layers_AE,
             neurons_fc=args.neurons_AE,
             hidden_activation=args.activation,
             out_layer_activation=None) # Encoder
print(net_E)
print("Encoder: number of parameters {} of layers {}".format(*net_E.num_parameters()))

# Discriminator
net_T = net_(inp_dim=X_train.shape[1] + Y_dim,
             out_dim=Y_dim,
             n_layers=args.num_layers_T,
             neurons_fc=args.neurons_AE,
             hidden_activation=args.activation,
             out_layer_activation=None) 

print(net_T)
print("Discriminator: number of parameters {} of layers {}".format(*net_T.num_parameters()))

# Wrapper around our model to handle parallel training
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net_D = nn.DataParallel(net_D)
  net_E = nn.DataParallel(net_E)
  net_T = nn.DataParallel(net_T)

net_D.to(device)
net_E.to(device)
net_T.to(device)

# Save diretory
model_dir = "Models/CADGMs_AE_{}x{}".\
    format(args.num_layers_AE, args.neurons_AE)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#%%
#================================ Training ===================================#

# Compute generator loss
def compute_G_loss(x, y, z, lam, beta): 
    # forward propagation 
    net_D.zero_grad()
    net_E.zero_grad()
    # Prior: p(z)
    z_prior = z
    # Decoder: p(y|x,z)
    y_pred = net_D(torch.cat((x, z_prior),dim=1))        
    # Encoder: q(z|x,y)
    z_encoder = net_E(torch.cat((x, y_pred),dim=1))
    # Discriminator output
    T_pred = net_T(torch.cat((x, y_pred),dim=1))
    # Estimated KL-divergence 
    KL = torch.mean(T_pred)
    # Entropic regularization
    log_q = -torch.mean(torch.square(z_prior-z_encoder))
    # Reconsturction Loss
    log_p = torch.mean(torch.square(y - y_pred))
    # Generator loss 
    loss_G = KL + (1.0-lam)*log_q + beta * log_p
    
    return loss_G, KL, log_p

# Compute discriminator loss
def compute_T_loss(x, y, z): 
    # forward propagation 
    net_D.zero_grad()
    net_T.zero_grad()
    
    # Prior: p(z)
    z_prior = z
    # Decoder: p(y|x,z)
    y_pred = net_D(torch.cat((x, z_prior),dim=1))               
    
    # Discriminator loss
    T_real = net_T(torch.cat((x, y),dim=1))
    T_fake = net_T(torch.cat((x, y_pred),dim=1))
    
    T_real = torch.sigmoid(T_real)
    T_fake = torch.sigmoid(T_fake)
    
    T_loss = -torch.mean(torch.log(1.0 - T_real + 1e-8) + \
                              torch.log(T_fake + 1e-8)) 
    
    return T_loss

def mse(y_true, y_pred, reduction='mean'):
    if reduction == 'mean':
        loss = torch.mean(torch.square(y_true - y_pred))
    elif reduction == 'sum':
        loss = torch.sum(torch.square(y_true - y_pred))
    return loss

def mae(y_true, y_pred, reduction='mean'):
    if reduction == 'mean':
        loss = torch.mean(torch.abs(y_true - y_pred))
    elif reduction == 'sum':
        loss = torch.sum(torch.abs(y_true - y_pred))
    return loss


def test():
    net_D.eval()
    X_batch, Y_batch = fetch_minibatch(X_test, Y_test, batch_size=args.batch_size)
    x = torch.LongTensor(X_batch).to(device)
    y = torch.FloatTensor(Y_batch).to(device)
    Z = torch.randn(x.shape[0], args.features, device=device)
    with torch.no_grad():
        y_pred = net_D(torch.cat((x, Z),dim=1))
            
        mse_ = mse(y, y_pred, reduction='mean').item()
        denominator = mse(y, torch.mean(y, axis=0), reduction='mean').item()
        mae_ = mae(y, y_pred, reduction='mean').item()
    
    
    rmse = np.sqrt(mse_)
    r2  = 1 - mse_ / denominator  
    return rmse, r2, mae_

optimizer_T = optim.Adam(net_T.parameters(), lr=args.lr)
optimizer_G = optim.Adam(list(net_D.parameters())+list(net_E.parameters()), lr=args.lr)

# How to schedule the learning rate
scheduler = lr_scheduler.MultiStepLR(optimizer_G, milestones=[0.5 * args.n_iterations, 0.75 * args.n_iterations], gamma=0.1)

print("Start training network")

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=10, min_delta=1e-6)


tic = time.time()
start_time = tic
save_loss_G, save_loss_T = [], []
r2_train, r2_test, rmse_train, rmse_test, mae_train, mae_test = [], [], [], [], [], []
loss_G = [] 
loss_T = []

for iter in range(1,args.n_iterations+1):
    net_D.train()
    net_E.train()
    net_T.train()
    
    # Fetch a batch
    X_batch, Y_batch = fetch_minibatch(X_train, Y_train, args.batch_size)
    x = torch.LongTensor(X_batch).to(device)
    y = torch.FloatTensor(Y_batch).to(device)
    Z = torch.randn(x.shape[0], args.features, device=device)
    
    optimizer_T.zero_grad()
    optimizer_G.zero_grad()
    
    #  minimize the loss
    for i in range(args.k1):
        T_loss = compute_T_loss(x, y, Z)
        # Backward Step 
        T_loss.backward()
        optimizer_T.step()
    
    for j in range(args.k2):
        G_loss, KL_loss, reconv = compute_G_loss(x, y, Z, args.lam, args.beta)
        # Backward Step 
        G_loss.backward()
        optimizer_G.step()
            
    
        
    loss_G.append(G_loss.item())
    loss_T.append(T_loss.item())
        
    # Compute metrics
    net_D.zero_grad()
    mse_ = mse(y, net_D(torch.cat((x, Z),dim=1)), reduction='mean').item()
    denominator = mse(y, torch.mean(y, axis=0), reduction='mean').item()
    mae_ = mae(y, net_D(torch.cat((x, Z),dim=1)), reduction='mean').item()
    
    #  Save the losses
    save_loss_G.append(loss_G)
    save_loss_T.append(loss_T)
    
    # Avarage over batches 
    rmse_  = np.sqrt(mse_)
    r2_    = 1 - mse_ / denominator
    
    # Check the model in the test data
    rmse_t, r2_t, mae_t = test()
    
    # save metrics
    rmse_train.append(rmse_)
    r2_train.append(r2_)
    mae_train.append(mae_)
    rmse_test.append(rmse_t)
    r2_test.append(r2_t)
    mae_test.append(mae_t)
    
    # Print
    if iter % args.log_interval == 0:
         
        elapsed = time.time() - start_time
        print('Iteration: %d, Generator_loss: %.2e, MSE: %.2e, Discriminator_loss: %.2e, Time: %.2f' % 
              (iter, KL_loss, reconv, T_loss, elapsed))
        print('Training RMSE:%.4e, Training R2-score:%.3f, Training MAE:%.3f' % (rmse_, r2_, mae_))
        print('Testing RMSE:%.4e, Testing R2-score:%.3f, Testing MAE:%.3f' % (rmse_t, r2_t, mae_t))
        start_time = time.time()
        
    scheduler.step()
    
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(rmse_, rmse_t)
        
    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(net_D.state_dict(), model_dir + "/Decoder.pt")
        torch.save(net_E.state_dict(), model_dir + "/Encoder.pt")
        torch.save(net_T.state_dict(), model_dir + "/Discriminator.pt")
        torch.save(save_loss_G, model_dir + "/loss_Generator.pt")
        torch.save(save_loss_T, model_dir + "/loss_Discriminator.pt")
        torch.save(rmse_train, model_dir + "/RMSE_train.pt")
        torch.save(r2_train, model_dir + "/R2_train.pt")
        torch.save(mae_train, model_dir + "/MAE_train.pt")
        torch.save(rmse_test, model_dir + "/RMSE_validation.pt")
        torch.save(r2_test, model_dir + "/R2_validation.pt")
        torch.save(mae_test, model_dir + "/MAE_validation.pt")
        break
    
    
    
    if iter == args.n_iterations:
        torch.save(net_D.state_dict(), model_dir + "/Decoder.pt")
        torch.save(net_E.state_dict(), model_dir + "/Encoder.pt")
        torch.save(net_T.state_dict(), model_dir + "/Discriminator.pt")
        torch.save(save_loss_G, model_dir + "/loss_Generator.pt")
        torch.save(save_loss_T, model_dir + "/loss_Discriminator.pt")
        torch.save(rmse_train, model_dir + "/RMSE_train.pt")
        torch.save(r2_train, model_dir + "/R2_train.pt")
        torch.save(mae_train, model_dir + "/MAE_train.pt")
        torch.save(rmse_test, model_dir + "/RMSE_validation.pt")
        torch.save(r2_test, model_dir + "/R2_validation.pt")
        torch.save(mae_test, model_dir + "/MAE_validation.pt")
    



tic2 = time.time()
print("Done training {} iterations in {} seconds"
      .format(args.n_iterations, tic2 - tic))

plt.figure(figsize=(8,6), dpi=150)
plt.plot(r2_train, 'b-', lw=2.0, label='Train')
plt.plot(r2_test, 'r-', lw=2.0, label='Test')
plt.ylim([0, 1.])
plt.box('True')
plt.grid('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'R$^2$-score',fontsize=18)
plt.xlabel(r'Number of Iterations',fontsize=18)
plt.legend(loc='best', frameon=False, prop={'size': 16, 'weight': 'extra bold'})
plt.savefig(model_dir + '/r2_score_training.jpg', bbox_inches='tight', dpi=150)

plt.figure(figsize=(8,6), dpi=150)
plt.plot(mae_train, 'b-', lw=2.0, label='Train')
plt.plot(mae_test, 'r-', lw=2.0, label='Test')
plt.box('True')
plt.grid('True')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel(r'MAE',fontsize=18)
plt.xlabel(r'Number of Iterations',fontsize=18)
plt.legend(loc='best', frameon=False, prop={'size': 16, 'weight': 'extra bold'})
plt.savefig(model_dir + '/mae_training.jpg', bbox_inches='tight', dpi=150)


#%%Prediction (generated samples as black lines, training data as red star)

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
    Z = torch.randn(X_star.shape[0], args.features, device=device)
    
    Y_star = sample_generator(X_star, Z) 
    # De-normalize outputs
    
    return Y_star

def scatter_plot(var1,var2, var3, var4, mae_t, mae_tt):
    min_data  = np.minimum(np.minimum(np.amin(var1),np.amin(var2)), np.minimum(np.amin(var3), np.amin(var4)))
    max_data = np.maximum(np.maximum(np.amax(var1),np.amax(var2)), np.maximum(np.amax(var3),np.amax(var4)))
    
    acc = (f'MAE (Train) = {mae_t:.2f}\n'
           f'MAE (Test) = {mae_tt:.2f}')
    
    plt.figure(figsize=(8,6),dpi=150)
    plt.scatter(np.reshape(var2,-1),np.reshape(var1,-1), 
                      s=50 ,marker='o', color = 'grey', alpha=0.5, label=r'Train')
    plt.scatter(np.reshape(var4,-1),np.reshape(var3,-1), 
                      s=50 ,marker='o', color = 'red', alpha=0.5, label=r'Test')
    xlim = plt.xlim(min_data, max_data)
    ylim = plt.ylim(min_data, max_data)
    plt.plot(xlim,ylim,'r--',lw= 2)
    # bbox = dict(boxstyle='round', fc='blue', ec='blue', alpha=0.15)
    # plt.text(130, 10, acc, fontsize=18, bbox=bbox, horizontalalignment='right')
    plt.grid('True')
    plt.box('True')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'Predicted CN',fontsize=18)
    plt.ylabel(r'Measured CN',fontsize=18)
    plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
    plt.title('')
    plt.savefig(model_dir + '/statistical_error_scatter.jpg', bbox_inches='tight', dpi=150)

def scatter_plot_CI(y_true, y_pred, var_pred, mae):
    min_data  = np.minimum(np.amin(y_true),np.amin(y_pred))
    max_data = np.maximum(np.amax(y_true),np.amax(y_pred))
    
    idx = np.argsort(y_pred)
    
    acc = (f'MAE = {mae:.2f}')
    
    plt.figure(figsize=(8,6),dpi=150)
    plt.scatter(np.reshape(y_pred,-1),np.reshape(y_true,-1), 
                      s=30 ,marker='o', c=y_true, cmap='jet')
    xlim = plt.xlim(min_data, max_data)
    ylim = plt.ylim(min_data, max_data)
    plt.plot(xlim,ylim,'k--',lw= 3)
    plt.fill_between(y_pred[idx], 
                     y_pred[idx] - 2*np.sqrt(var_pred[idx]), 
                     y_pred[idx] + 2*np.sqrt(var_pred[idx]), 
                     color='grey',
                     alpha=0.25, label='95% C.I')
    
    bbox = dict(boxstyle='round', fc='blue', ec='blue', alpha=0.15)
    plt.text(130, 10, acc, fontsize=18, bbox=bbox, horizontalalignment='right')
    plt.grid('True')
    plt.box('True')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'Predicted CN',fontsize=18)
    plt.ylabel(r'Measured CN',fontsize=18)
    plt.legend(loc='best', frameon=False, prop={'size': 18, 'weight': 'extra bold'})
    plt.title('')
    plt.savefig(model_dir + '/scatter_CI.jpg', bbox_inches='tight', dpi=150)


# whole data
N_samples = 1000
X_star = feat
X_star = torch.LongTensor(X_star).to(device)
samples_mean= np.zeros((X_star.shape[0], N_samples))
for i in range(0, N_samples): 
    samples_mean[:,i:i+1] = generate_sample(X_star).detach().cpu().numpy() 
    
# Compute mean and variance of the prediction as function of x
mu_pred = np.mean(samples_mean, axis = 1) * std_CN + mu_CN  
var_pred = np.var(samples_mean, axis = 1) * std_CN + mu_CN 

CN = CN.astype(str)
mask = CN != 'nan'
Y_star = np.zeros(len(CN))
for idx in range(len(CN)):
    mask = CN[idx] != 'nan'
    Y_star[idx] = np.mean(CN[idx, mask].astype(float)) 

# scatter ci plot
mae = mean_absolute_error(Y_star, mu_pred)
scatter_plot_CI(Y_star, mu_pred, var_pred, mae)

# train
X_train = torch.LongTensor(X_train).to(device)
samples_mean_train = np.zeros((X_train.shape[0], N_samples))
for i in range(0, N_samples): 
    samples_mean_train[:,i:i+1] = generate_sample(X_train).detach().cpu().numpy()  
    
# Compute mean and variance of the prediction as function of x
mu_pred_train = np.mean(samples_mean_train, axis = 1) * std_CN + mu_CN    
var_pred_train = np.var(samples_mean_train, axis = 1) * std_CN + mu_CN 

# "Diagnostics: How do you know if the fit is good?"
mask = Y_train != 'nan'
Y_star = np.zeros(len(Y_train))
for idx in range(len(Y_train)):
    mask = Y_train[idx] != 'nan'
    Y_star[idx] = np.mean(Y_train[idx, mask].astype(float)) 

Y_train = Y_train.astype(float) * std_CN + mu_CN
Y_star_train = Y_star * std_CN + mu_CN

mape_train = mean_absolute_percentage_error(Y_star_train, mu_pred_train)
mae_train = mean_absolute_error(Y_star_train, mu_pred_train)
rmse_train = root_mean_squared_error(Y_star_train, mu_pred_train)
r2_train = r2_score(Y_star_train, mu_pred_train)

print("Default selection Train MAE:",mae_train)
print("Default selection R2-score:",r2_train)

# test
X_test = torch.LongTensor(X_test).to(device)
samples_mean_test = np.zeros((X_test.shape[0], N_samples))
for i in range(0, N_samples):
 	samples_mean_test[:,i:i+1] = generate_sample(X_test).detach().cpu().numpy() 

# Compute mean and variance of the prediction as function of x
mu_pred_test = np.mean(samples_mean_test, axis = 1) * std_CN + mu_CN
var_pred_test = np.var(samples_mean_test, axis = 1) * std_CN + mu_CN

# "Diagnostics: How do you know if the fit is good?"
mask = Y_test != 'nan'
Y_star = np.zeros(len(Y_test))
for idx in range(len(Y_test)):
    mask = Y_test[idx] != 'nan'
    Y_star[idx] = np.mean(Y_test[idx, mask].astype(float)) 

Y_test = Y_test.astype(float) * std_CN + mu_CN
Y_star_test = Y_star * std_CN + mu_CN

mape_test = mean_absolute_percentage_error(Y_star_test, mu_pred_test)
mae_test = mean_absolute_error(Y_star_test, mu_pred_test)
rmse_test = root_mean_squared_error(Y_star_test, mu_pred_test)
r2_test = r2_score(Y_star_test, mu_pred_test)

print("Default selection Test MAE:",mae_test)
print("Default selection R2-score:",r2_test)


# creata a dataframe
d = {'Train': [mape_train, mae_train, rmse_train, r2_train], #
      'Test': [mape_test, mae_test, rmse_test, r2_test]}

metrics = pd.DataFrame(data=d, index=['MAPE', 'MAE', 'RMSE', 'R2'])

metrics.to_excel(model_dir + '/metrics.xlsx')

# Scatter plot
scatter_plot(Y_star_train, mu_pred_train, Y_star_test, mu_pred_test, mae_train, mae_test)

# idx_test = np.argsort(Y_star_test)
# idx_train = np.argsort(Y_star_train)

# plt.figure(figsize=(16, 6), dpi=150)  
# plt.errorbar(np.arange(idx_train[::12].shape[0]), 
#               mu_pred_train[idx_train[::12]], 
#               2*np.sqrt(var_pred_train[idx_train[::12]]), 
#               ecolor='c',
#               marker='s',
#               cmarkersize=10,
#               capsize=10,  
#               label = r'$\mu \pm 2\sigma$', 
#               linestyle='none')
# plt.plot(np.arange(idx_train[::12].shape[0]), Y_train[idx_train[::12]], 'ks', markersize = 10)
# plt.xticks(np.arange(idx_train[::12].shape[0]), fuel_train[idx_test[::12]], fontsize=20, rotation=90)
# plt.yticks(fontsize=24)
# #plt.xlabel('fuels',fontsize=18)
# plt.ylabel('Cetane Number',fontsize=24)
# plt.legend(loc='upper left', frameon=False, prop={'size': 26, 'weight': 'extra bold'})
# plt.savefig(model_dir + '/train_uq_analysis.jpg', bbox_inches='tight', dpi=150)


# plt.figure(figsize=(16, 6), dpi=150)  
# plt.errorbar(np.arange(idx_test[::5].shape[0]), 
#               mu_pred_test[idx_test[::5]], 
#               2*np.sqrt(var_pred_test[idx_test[::5]]), 
#               ecolor='c',
#               marker='s',
#               markersize=10,
#               capsize=10,  
#               label = r'$\mu \pm 2\sigma$', 
#               linestyle='none')
# plt.plot(np.arange(idx_test[::5].shape[0]), Y_test[idx_test[::5]], 'rv', markersize = 10)
# plt.xticks(np.arange(idx_test[::5].shape[0]), fuel_test[idx_test[::5]], fontsize=24, rotation=90)
# plt.yticks(fontsize=24)
# #plt.xlabel('fuels', fontsize=18)
# plt.ylabel('Cetane Number',fontsize=24)
# plt.legend(loc='upper left', frameon=False, prop={'size': 26, 'weight': 'extra bold'})
# plt.savefig(model_dir + '/test_uq_analysis.jpg', bbox_inches='tight', dpi=150)
