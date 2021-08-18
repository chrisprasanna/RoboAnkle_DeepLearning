"""
Created on Wed Apr 21 12:23:55 2021

Prosthetic Ankle Torque Predictor
Optimization Script for all Networks 

This script is the entry point of the full hyperparameter optimization and 
neural network training procedure. The role of this script is to load the 
dataset, specify the hyperparameter search ranges, initialize neural network 
training, evaluate the optimized networks, and save the results.  

@author: Chris Prasanna
"""

#%% Imports

import torch
import torch.nn as nn

import os
import time

from DL_functions import loadmat
from Optuna_functions import optimize_hyperparams, get_dataLoaders
from Test_functions import test_network, visualize_results

#%% Load and Organize Data

# Load Data from .mat file
print('Loading Data...')
cwd = os.getcwd()
data_path = os.path.dirname(cwd)
matfile = os.path.join(data_path, 'JR_data_ankleTorque.mat')
# matfile = r'JR_data_ankleTorque.mat'
matdata = loadmat(matfile)
Data = matdata['Data']
print('Loading Complete')

# Size of Data
dims = Data['Features'].shape
num_trials = dims[0]
num_timestepsPerTrial = dims[1]
num_features = dims[2]

active_features = Data['activeFeatures']
passive_features = Data['passiveFeatures']
num_activeTrials = active_features.shape[0]
num_passiveTrials = passive_features.shape[0]

# Inputs and Outputs
features = Data['Features']
responses = Data['Responses']

#%% Hyperparameters and Optuna Options

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DNN Hyperparameters
input_size = num_features
output_size = 1
step_size = 1
fs = 120 # sampling freq 

# Optimizer Hyperparameters
lr_step_size = 3           # number of epochs before reducing learning rate
lr_patience = 3
min_lr = 1e-5
amsgrad = False

# Training Hyperparameters
num_epochs = 1000
batch_size = num_trials

# Loss Function
criterion = nn.MSELoss()

# Optuna 
optuna_trials = 500
optuna_timeout = None

# Hyperparameters to Optimize Using Optuna
# - Hidden size
# - Number of Layers
# - Initial Learning Rate
# - STD of Gaussian Noise
# - Dropout Factor
# - Type of Optimizer (Adam, AdamW, SGD+momentum, RMSProp, etc.)
# - Weight decay / L2 Regularization
# - Learning Rate decrease factor 
# - Sequence Length

# hidden_size = 256       # [4,8,16,32,64,128,256,512]
# num_layers = 2
# learning_rate = 0.001   # inital LR
# noiseSTD = 0.2
# dropout = 0.2
# weight_decay = 0.05     # L2Regularization (default = 0); tested: 1e-5, 0.05
# gamma = 0.5             # learning rate decrease factor
# sequence_length = 10

#%% Save Paths

timestr = time.strftime("%Y%m%d-%H%M%S")
directory = f'{timestr[:-2]}'

os.makedirs(os.path.join(cwd, 'Results',directory)) 

os.makedirs(os.path.join(cwd, 'Results',directory,'FFN')) 
os.makedirs(os.path.join(cwd, 'Results',directory,'GRU')) 
# os.makedirs(os.path.join(cwd, 'Results',directory,'Transformer')) 
os.makedirs(os.path.join(cwd, 'Results',directory,'DA-RNN')) 

PATH = os.path.join(cwd,'Results',directory) 


#%% Create a Dictionary Object of Constants to pass through Functions

constants = {'device':device,
             'results path':PATH,
             'input size':input_size,
             'output size':output_size,
             'learning rate scheduler delay':lr_step_size,
             'learning rate scheduler patience':lr_patience, 
             'minimum learning rate':min_lr, 
             'number of walking trials':num_trials,
             'number of timesteps per trial':num_timestepsPerTrial,
             'loss function':criterion,
             'max number of epochs':num_epochs,
             'number of optuna trials':optuna_trials,
             'optuna timeout':optuna_timeout,
             'dataset': Data
             }

# Independent Variables
#   - model name
#   - current optuna trial
#   - model
#   - scheduler
#   - optimizer
#   - sequence length

#%% Optimize Hyperparameters for all NNs

print(" *** Optimizing the Networks...")
 
FFN_model, FFN_trial = optimize_hyperparams('FFN', constants)

GRU_model, GRU_trial = optimize_hyperparams('GRU', constants)

# Transformer_model, Transformer_trial = optimize_hyperparams('Transformer', constants)

DARNN_model, DARNN_trial = optimize_hyperparams('DA-RNN', constants)

#%% Test Trained Network

print(" *** Testing the Networks...")

sequence_length_FFN = FFN_trial.params['sequence_length']
sequence_length_GRU = GRU_trial.params['sequence_length']
# sequence_length_Transformer = Transformer_trial.params['sequence_length']
sequence_length_DARNN = DARNN_trial.params['sequence_length']

# FFN
train_loader_FFN, val_loader_FFN, test_loader_FFN = get_dataLoaders(FFN_trial, sequence_length_FFN, constants)
target_FFN, pred_FFN, RMSE_FFN, test_loss_FFN, pcc_FFN = test_network(test_loader_FFN, FFN_model, 
                                                         'FFN', criterion, PATH, device)

# GRU
train_loader_GRU, val_loader_GRU, test_loader_GRU = get_dataLoaders(GRU_trial, sequence_length_GRU, constants)
target_GRU, pred_GRU, RMSE_GRU, test_loss_GRU, pcc_GRU = test_network(test_loader_GRU, GRU_model, 
                                                         'GRU', criterion, PATH, device)

# # Transformer
# train_loader_Transformer, val_loader_Transformer, test_loader_Transformer = get_dataLoaders(Transformer_trial, sequence_length_Transformer, constants)
# target_Transformer, pred_Transformer, RMSE_Transformer, test_loss_Transformer, pcc_Transformer = test_network(test_loader_Transformer, 
#                                                                                              Transformer_model, 'Transformer', 
#                                                                                              criterion, PATH)

# Da-RNN
train_loader_DARNN, val_loader_DARNN, test_loader_DARNN = get_dataLoaders(DARNN_trial, sequence_length_DARNN, constants)
target_DARNN, pred_DARNN, RMSE_DARNN, test_loss_DARNN, pcc_DARNN = test_network(test_loader_DARNN, DARNN_model, 
                                                         'DA-RNN', criterion, PATH, device)

#%% Save Models

print(" *** Saving the Networks...")

# FFN
filename = f'FFN NN {timestr[:-2]}.pt'
FFN_PATH = os.path.join(PATH,'FFN',filename) 
net = FFN_model.state_dict()
torch.save(net, FFN_PATH)

# GRU
h = GRU_model.init_hidden(batch_size = 1)
filename = f'GRU NN {timestr[:-2]}.pt'
GRU_PATH = os.path.join(PATH,'GRU',filename) 
net = GRU_model.state_dict()
torch.save(net, GRU_PATH)

# Transformer
# filename = f'Transformer NN {timestr[:-2]}.pt'
# Transformer_PATH = os.path.join(PATH,'Transformer',filename) 
# net = Transformer_model.state_dict()
# torch.save(net, Transformer_PATH)

# DA-RNN
filename = f'DA-RNN NN {timestr[:-2]}.pt'
DARNN_PATH = os.path.join(PATH,'DA-RNN',filename) 
net = DARNN_model.state_dict()
torch.save(net, DARNN_PATH)


#%% Visualize Test Data Results

print(" *** Plotting the Test Results...")

# FFN
percentFit_FFN, percentError_FFN = visualize_results(FFN_model, 'FFN', train_loader_FFN, 
                                             val_loader_FFN, test_loader_FFN, num_trials, 
                                             target_FFN, pred_FFN, PATH, fs, device)

# GRU
percentFit_GRU, percentError_GRU = visualize_results(GRU_model, 'GRU', train_loader_GRU, 
                                             val_loader_GRU, test_loader_GRU, num_trials, 
                                             target_GRU, pred_GRU, PATH, fs, device)

# # Transformer
# percentFit_Transformer, percentError_Transformer = visualize_results(Transformer_model, 'Transformer', 
#                                                      train_loader_Transformer, val_loader_Transformer, 
#                                                      test_loader_Transformer, num_trials, target_Transformer, 
#                                                      pred_Transformer, PATH, fs, device)

# DA-RNN
percentFit_DARNN, percentError_DARNN = visualize_results(DARNN_model, 'DA-RNN', train_loader_DARNN, 
                                             val_loader_DARNN, test_loader_DARNN, num_trials, 
                                             target_DARNN, pred_DARNN, PATH, fs, device)

print("\n\n*** Finished ***")
