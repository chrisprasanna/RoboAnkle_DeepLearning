"""
Created on Wed Apr 21 12:23:55 2021

Prosthetic Ankle Torque Predictor
Optimization Script for all Networks 

@author: Chris Prasanna
"""

#%% Imports

import torch
import torch.nn as nn
from scipy import signal
import numpy as np

import os
import time
import sys

from DL_functions import loadmat
from Optuna_functions import optimize_hyperparams, get_dataLoaders
from Test_functions import test_network, visualize_results, fitted_histogram

from GPUtil import showUtilization as gpu_usage

#%% Load and Organize Data

# Clear cuda memory
torch.cuda.empty_cache()
print("Initial GPU Usage")
gpu_usage() 

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
fsNew = 30 # data rate for sub-sampling
prediction_horizon = 1 # set as 1 for simulation

# Optimizer Hyperparameters
lr_step_size = 3           # number of epochs before reducing learning rate
lr_patience = 3
min_lr = 1e-5
amsgrad = False

# Loss Function
criterion = nn.MSELoss()

# Training Hyperparameters
num_epochs = 1000
optuna_trials = 500
batch_size = num_trials
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
os.makedirs(os.path.join(cwd, 'Results',directory,'DA-RNN')) 

PATH = os.path.join(cwd,'Results',directory) 

#%% Resample Time Series Data

fsNew = 30 # data rate for sub-sampling
fsRatio = fsNew / fs
num_timestepsPerTrial = int(num_timestepsPerTrial * fsRatio)

for key in Data:
     Data[key] = signal.resample(Data[key], num_timestepsPerTrial, axis=1)

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
             'dataset': Data, 
             'pred_horizon': prediction_horizon
             }

# Independent Variables
#   - model name
#   - current optuna trial
#   - model
#   - scheduler
#   - optimizer
#   - sequence length

#%% Optimize Hyperparameters for all NNs

# remove pickle files
dir_name = os.getcwd()
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".pickle"):
        os.remove(os.path.join(dir_name, item))

print(" *** Optimizing the Networks...")

torch.cuda.empty_cache() 
FFN_model, FFN_trial = optimize_hyperparams('FFN', constants)

torch.cuda.empty_cache()
GRU_model, GRU_trial = optimize_hyperparams('GRU', constants)

torch.cuda.empty_cache()
DARNN_model, DARNN_trial = optimize_hyperparams('DA-RNN', constants)

#%% Test Trained Network

print(" *** Testing the Networks...")

sequence_length_FFN = FFN_trial.params['sequence_length']
sequence_length_GRU = GRU_trial.params['sequence_length']
sequence_length_DARNN = DARNN_trial.params['sequence_length']

# FFN
train_loader_FFN, val_loader_FFN, test_loader_FFN = get_dataLoaders(FFN_trial, sequence_length_FFN, 
                                                                    constants, train_bool=False)
target_FFN, pred_FFN, RMSE_FFN, test_loss_FFN, pcc_FFN = test_network(test_loader_FFN, FFN_model, 
                                                         'FFN', criterion, PATH, device)

# GRU
train_loader_GRU, val_loader_GRU, test_loader_GRU = get_dataLoaders(GRU_trial, sequence_length_GRU, 
                                                                    constants, train_bool=False)
target_GRU, pred_GRU, RMSE_GRU, test_loss_GRU, pcc_GRU = test_network(test_loader_GRU, GRU_model, 
                                                         'GRU', criterion, PATH, device)
                                                                                            

# Da-RNN
train_loader_DARNN, val_loader_DARNN, test_loader_DARNN = get_dataLoaders(DARNN_trial, sequence_length_DARNN, 
                                                                    constants, train_bool=False)
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
                                             target_FFN, pred_FFN, PATH, fsNew, device)

# GRU
percentFit_GRU, percentError_GRU = visualize_results(GRU_model, 'GRU', train_loader_GRU, 
                                             val_loader_GRU, test_loader_GRU, num_trials, 
                                             target_GRU, pred_GRU, PATH, fsNew, device)

# DA-RNN
percentFit_DARNN, percentError_DARNN = visualize_results(DARNN_model, 'DA-RNN', train_loader_DARNN, 
                                             val_loader_DARNN, test_loader_DARNN, num_trials, 
                                             target_DARNN, pred_DARNN, PATH, fsNew, device)

# All Models
fitted_histogram(target_FFN, pred_FFN, target_GRU, pred_GRU, target_DARNN, pred_DARNN, PATH)

print("\n\n*** Finished ***")
