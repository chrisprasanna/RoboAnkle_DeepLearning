# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:17 2021

Objective functions for all NNs
Includes getting the model and datasets

@author: cpras
"""

#%% Imports

from NN_classes import FFN, GRU, TransformerModel, DARNN
from DL_classes import CustomTensorDataset
# from Train_functions import train_FFN, train_GRU, train_Transformer, train_DARNN
from main_Train import train_PAFP

import torch
import torch.nn as nn
import torch.optim as optim  # ADAM, SGD, etc
# import torch.utils.data.SubsetRandomSampler as SubsetRandomSampler

import optuna
from optuna.trial import TrialState
import pickle as pickle # import pickle5 as pickle

import numpy as np
import pandas as pd
import os
import time as timer

#%% Get Model

def define_model(trial, model_name, sequence_length, device, constants):
    
    # Unpack constants
    input_size = constants['input size']
    output_size = constants['output size']
    
    # Which model type are we optimizing?
    if model_name == 'FFN':
        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_size_power = trial.suggest_int("hidden_size_power", 4, 11)
        hidden_size = 2**hidden_size_power
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  
        
        model = FFN(input_size, hidden_size, output_size, num_layers, sequence_length, dropout)
        
    elif model_name == 'GRU':
        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_size_power = trial.suggest_int("hidden_size_power", 4, 11)
        hidden_size = 2**hidden_size_power
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  
        
        model = GRU(input_size, hidden_size, output_size, num_layers, sequence_length, dropout, device)     
        
    elif model_name == 'DA-RNN':
        hidden_size_power = trial.suggest_int("hidden_size_power", 4, 11)
        hidden_size = 2**hidden_size_power
        P_power = trial.suggest_int("decoder_size_power", 4, 7)
        P = 2**P_power
        
        model = DARNN(input_size, hidden_size, P, sequence_length, device)
        
    else:
        print('Model Name is not one of the four options!!')
        print('Must be FFN, GRU, Transformer, or DA-RNN')
        model = []
        
    return model

#%% Get Datasets

def get_dataLoaders(trial, sequence_length, constants, train_bool):
    
    # Unpack constants
    input_size = constants['input size']
    output_size = constants['output size']
    num_trials = constants['number of walking trials']
    num_timestepsPerTrial = constants['number of timesteps per trial']
    Data = constants['dataset']
    prediction_horizon = constants['pred_horizon']
    
    # Temporal Processing

    print('Processing Time Series Data...')
    
    # Inputs and Outputs
    features = Data['Features']
    responses = Data['Responses']
    
    # Pre-allocate
    X = np.zeros((num_trials, num_timestepsPerTrial, sequence_length, input_size))
    y = np.zeros((num_trials, num_timestepsPerTrial, sequence_length, output_size))
    target = np.zeros((num_trials, num_timestepsPerTrial, output_size))
    
    for n in range(num_trials):
    
        data_x = pd.DataFrame(features[n,:,:])
        data_y = pd.DataFrame(responses[n,:])
        
        for i, name in enumerate(list(data_x.columns)):
            for j in range(sequence_length):
                X[n, :, j, i] = data_x[name].shift(sequence_length - j - 1).fillna(method="bfill")
                
        for j in range(sequence_length):
            y[n,:,j,0] = data_y[0].shift(sequence_length - j - 1).fillna(method="bfill")
                
        # prediction_horizon = 1
        target[n,:,0] = data_y[0].shift(-prediction_horizon).fillna(method="ffill").values
    
        
    X = X[:,sequence_length:-1]
    y = y[:,sequence_length:-1]
    target = target[:,sequence_length:-1]
    
    # Split Dataset - Train, Val, Test
    
    print('Splitting Data into Training, Validation, and Test Sets...')
    
    timesteps = num_timestepsPerTrial - sequence_length - 1
    
    tr_percent = 0.70
    val_percent = 0.15
    test_percent = 0.15
    
    train_length = int(np.round(timesteps*tr_percent))
    test_length = int(np.floor(timesteps*test_percent))
    val_length = timesteps - train_length - test_length
    
    X_train = X[:,:train_length]
    X_val = X[:,train_length:train_length+val_length]
    X_test = X[:,train_length+val_length:]
    target_train = target[:,:train_length]
    target_val = target[:,train_length:train_length+val_length]
    target_test = target[:,train_length+val_length:]
    
    # Data Scaling
    
    print('Scaling Data from [0,1]...')
    
    X_train_max = X_train.max(axis=(0,1,2))
    X_train_min = X_train.min(axis=(0,1,2))
    target_train_max = target_train.max(axis=(0,1))
    target_train_min = target_train.min(axis=(0,1))
    
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)
    
    # target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
    # target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
    # target_test = (target_test - target_train_min) / (target_train_max - target_train_min)
    
    # Convert to Tensors
    
    print('Create Pytorch Data Loaders...')
    
    X_train_t = torch.Tensor(X_train)
    X_val_t = torch.Tensor(X_val)
    X_test_t = torch.Tensor(X_test)
    target_train_t = torch.Tensor(target_train)
    target_val_t = torch.Tensor(target_val)
    target_test_t = torch.Tensor(target_test)
    
    # Add Guassian Noise to Training and Validation Inputs
    if train_bool:
        noiseSTD = trial.suggest_float("noiseSTD", 0.1, 1, log=False)
    else:
        noiseSTD = 0
    
    Xtrain_noisy = X_train_t  + noiseSTD*torch.randn(X_train_t.size())
    Xval_noisy = X_val_t + noiseSTD*torch.randn(X_val_t.size())
    
    ## Create Data Loaders
    if train_bool:
        
        # Suggest a batch size for optimization
        batch_size_power = trial.suggest_int("batch_size_power", 4, 8)
        batch_size = 2**batch_size_power
        
        # (timesteps, trials, sequence, features)
        # This way we are dividing batches by timesteps and not trials
        Xtrain_noisy = Xtrain_noisy.permute(1,0,2,3)
        # X_val_t = X_val_t.permute(1,0,2,3)
        # X_test_t = X_test_t.permute(1,0,2,3)
        target_train_t = target_train_t.permute(1,0,2)
        # target_val_t = target_val_t.permute(1,0,2)
        # target_test_t = target_test_t.permute(1,0,2)
        
        # Create datasets
        train_dataset = CustomTensorDataset([Xtrain_noisy, target_train_t])
        val_dataset = CustomTensorDataset([X_val_t, target_val_t])
        test_dataset = CustomTensorDataset([X_test_t, target_test_t])

        # dataloader options 
        train_shuffle = False   # data reshuffled at every epoch
        batch_drop = True       # drop the last incomplete batch, if the dataset size is not divisible by the batch size                           
        # sampler = SubsetRandomSampler()
        
        # Create Dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
                                                   shuffle=train_shuffle, drop_last=batch_drop)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
        
    else:
        
        # Create Datasets (trials, timesteps, sequence, features)
        train_dataset = CustomTensorDataset([X_train_t, target_train_t])
        val_dataset = CustomTensorDataset([X_val_t, target_val_t])
        test_dataset = CustomTensorDataset([X_test_t, target_test_t])                              
        
        # Create Dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    
    
    return train_loader, val_loader, test_loader

#%% Objective

def objective(trial, model_name, constants):
    
    # Empty GPU Cache
    torch.cuda.empty_cache()
    
    # Unpack constants
    device = constants['device']
    lr_step_size = constants['learning rate scheduler delay']
    lr_patience = constants['learning rate scheduler patience']
    lr_min = constants['minimum learning rate']
    criterion = constants['loss function']
    num_epochs = constants['max number of epochs']
    
    # Start timer
    start = timer.time()
    
    # Suggest a sequence length
    sequence_length = trial.suggest_int("sequence_length", 2, 25, log=False) # one step ~= 150 points
    
    # Generate the Model
    model = define_model(trial, model_name, sequence_length, device, constants).to(device)
    
    # Get Training Dataset with Added Noise
    train_loader, val_loader, test_loader = get_dataLoaders(trial, sequence_length, constants, train_bool=True)
    
    # Optimizer Options
    optimizer_name = "AdamW"
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=False)
    gamma = trial.suggest_float("gamma", 0.1, 0.9, log=False)
    
    # Generate the Optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Generate the Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, 
                                                 patience=lr_patience, cooldown=lr_step_size, 
                                                 min_lr=lr_min, verbose=True)

    # Prints Params
    print("\nTrain {} Trial {} with Params: ".format(model_name, trial.number))
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print('\n')
    
    # Train
    MSE = train_PAFP(trial, model_name, model, scheduler, optimizer, criterion,  
                        train_loader, val_loader, num_epochs, device)
    
    # Train time
    train_time = timer.time() - start
    
    # Objective function
    w_mse = 1 # weight associated with validation MSE
    w_time = 0 # weight associated with training time
    J = (w_mse*MSE) + (w_time*train_time) # Cost
    
    return J

#%% Optimization

def optimize_hyperparams(model_name, constants):
    
    # Unpack constants
    PATH = constants['results path']
    optuna_trials = constants['number of optuna trials']
    optuna_timeout = constants['optuna timeout']
    
    # Create an optimization project
    print('\n')
    study = optuna.create_study(study_name = "Optimize_{}".format(model_name), direction="minimize")
    
    # Optimize the network
    study.optimize(lambda trial: objective(trial, model_name, constants), 
                   n_trials=optuna_trials, timeout=optuna_timeout, gc_after_trial=True)
    
    # Summary Report
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    filename = os.path.join(PATH, model_name, 'Optimization_Results.txt')
    print("{} Study statistics: ".format(model_name), file=open(filename, "a"))
    print("  Number of finished trials: ", len(study.trials), file=open(filename, "a"))
    print("  Number of pruned trials: ", len(pruned_trials), file=open(filename, "a"))
    print("  Number of complete trials: ", len(complete_trials), file=open(filename, "a"))
    
    print("\n{} Best trial:".format(model_name), file=open(filename, "a"))
    trial = study.best_trial
    
    print("\n  Validation Obj. Func. Value: ", trial.value, file=open(filename, "a"))
    
    print("\n  {} Params: ".format(model_name), file=open(filename, "a"))
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value), file=open(filename, "a"))
    
    # Load the best model
    pickle_file = "{}_{}.pickle".format(model_name, study.best_trial.number)
    with open(pickle_file, "rb") as fin:
        best_model = pickle.load(fin)
        
    return best_model, trial
