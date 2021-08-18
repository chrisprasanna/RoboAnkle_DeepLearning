# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:57:57 2021

Functions to train each NN, which includes a validation dataset

@author: cpras
"""

#%% Imports

from DL_classes import EarlyStopping

import torch
import numpy as np

import optuna
import pickle as pickle # import pickle5 as pickle

from tqdm import tqdm

import GPUtil
from GPUtil import showUtilization as gpu_usage

#%% Train FFN

def train_FFN(trial, model, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    train_length = train_loader.dataset.data.shape[1]
    val_length = val_loader.dataset.data.shape[1]
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='FFN_checkpoint.pt')
    
    for epoch in range(num_epochs):
        
        ###################
        # Train the model #
        ###################
        model.train() # prep model for training
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for (idx, batch) in loop: # in enumerate(train_loader)
        
            batch_x = batch[0].squeeze().to(device)
            batch_y = batch[1].squeeze().to(device)
            
            # Concatenate input vector
            # x = torch.flatten(batch_x, start_dim=1)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            out = model(batch_x.float())
            
            # calculate the loss
            yTrue = batch_y.float() # seq2seq
            loss = criterion(out.squeeze(), yTrue) # if seq to one, only last timestep ([:,-1]) is used for loss 
            
            # record training loss
            train_losses.append(loss.item())
            
            # Backward
            optimizer.zero_grad() # clear the gradients of all optimized variables
            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
        
            # Gradient Descent or ADAM step
            optimizer.step()  # update the weights
            
            # Update Progress Bar
            loop.set_description(f"FFN Trial {trial.number} Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
            # delete intermediate values
            del batch_x, batch_y, out, yTrue, loss
            torch.cuda.empty_cache()
            
        ######################    
        # Validate the model #
        ######################
        model.eval() # prep model for evaluation
        
        with torch.no_grad():
            
            for (idx, batch) in enumerate(val_loader):
                
                batch_x = batch[0].squeeze().to(device)
                batch_y = batch[1].squeeze().to(device)
                
                # Concatenate input vector
                # x = torch.flatten(batch_x, start_dim=1)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                out = model(batch_x.float())
                
                # calculate the loss
                yTrue = batch_y.float() # seq2seq
                loss = criterion(out.squeeze(), yTrue)
                
                # record validation loss
                valid_losses.append(loss.item())
                
                # delete intermediate values
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(num_epochs))
        print()
        print_msg = (f'FFN [{epoch+1:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                     f'train_loss: {train_loss:.5f} ' + 
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)    
        print()
        
        # Update the learning rate
        scheduler.step(valid_loss)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        ## Optuna
        MSE = valid_loss
        trial.report(MSE, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # GPU Memory 
        print("\nGPU Usage after Train/Val Epoch")
        gpu_usage()
    
    ## Save a trained model to a file.
    with open("FFN_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    return MSE


#%% Train GRU

def train_GRU(trial, model, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    # Timesteps / batch size
    train_length = train_loader.dataset.data.shape[0]
    val_length = val_loader.dataset.data.shape[0]
    
    # Init hidden state
    h = model.init_hidden(batch_size = train_length)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='GRU_checkpoint.pt')
    
    for epoch in range(num_epochs):
    
        ###################
        # Train the model #
        ###################
        
        model.train() # prep model for training
        
        # ***************
        # Note: include code that turns gradients on for backprop
        # >> is_train = True
        # >> with torch.set_grad_enabled(is_train):
        # OR if that doesn't work
        # >> with torch.enable_grad():
        # ***************
        h = model.init_hidden(batch_size = train_length)
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for (idx, batch) in loop: # in enumerate(train_loader)
        
            batch_x = batch[0].squeeze().to(device)
            batch_y = batch[1].squeeze().to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            out,h = model(batch_x.float(),h)
            out = out[:,-1,0]
            
            # calculate the loss
            yTrue = batch_y.float() # seq2seq
            loss = criterion(out.squeeze(), yTrue) # if seq to one, only last timestep ([:,-1]) is used for loss 
            
            # record training loss
            train_losses.append(loss.item())
            
            # Backward
            optimizer.zero_grad() # clear the gradients of all optimized variables
            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
        
            # Gradient Descent or ADAM step
            optimizer.step()  # update the weights
            
            # detach hidden in between batches 
            h = h.detach()
            
            # Update Progress Bar
            loop.set_description(f"GRU Trial {trial.number} Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
            # ****************
            # NOTE: delete intermediate variables to free up GPU memory
            # >> del batch_x, batch_y, out, yTrue, loss
            # ****************
            del batch_x, batch_y, out, yTrue, loss
            torch.cuda.empty_cache()
            
        ######################    
        # Validate the model #
        ######################
        
        model.eval() # prep model for evaluation
        
        # ***************
        # Note: include code that turns gradients off to save memory
        # >> is_train = False
        # >> with torch.set_grad_enabled(is_train):
        # OR if that doesn't work
        # >> with torch.no_grad():
        # ***************
        with torch.no_grad():
        
            h_val = model.init_hidden(batch_size = val_length)
            for (idx, batch) in enumerate(val_loader):
                
                batch_x = batch[0].squeeze().to(device)
                batch_y = batch[1].squeeze().to(device)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                out,h_val = model(batch_x.float(),h_val)
                out = out[:,-1,0]
                
                # calculate the loss
                yTrue = batch_y.float() # seq2seq
                loss = criterion(out.squeeze(), yTrue)
                
                # record validation loss
                valid_losses.append(loss.item())
                
                # Delete intermediate variables
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()
        
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(num_epochs))
        print()
        print_msg = (f'GRU [{epoch+1:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                     f'train_loss: {train_loss:.5f} ' + 
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)    
        print()
        
        # Update the learning rate
        scheduler.step(valid_loss)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        ## Optuna
        MSE = valid_loss
        trial.report(MSE, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # GPU Memory 
        print("\nGPU Usage after Train/Val Epoch")
        gpu_usage()
    
    ## Save a trained model to a file.
    with open("GRU_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
        
    return MSE
    
#%% Train Transformer

def train_Transformer(trial, model, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    train_length = train_loader.dataset.data.shape[1]
    val_length = val_loader.dataset.data.shape[1]
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='Transformer_checkpoint.pt')
    
    for epoch in range(num_epochs):
        
        ###################
        # Train the model #
        ###################
        model.train() # prep model for training
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for (idx, batch) in loop: # in enumerate(train_loader)
        
            batch_x = batch[0].squeeze().to(device)
            batch_y = batch[1].squeeze().to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            out = model(batch_x.float())
            out = out[:,-1,0]
            
            # calculate the loss
            yTrue = batch_y.float() # seq2seq
            loss = criterion(out.squeeze(), yTrue) # if seq to one, only last timestep ([:,-1]) is used for loss 
            
            # record training loss
            train_losses.append(loss.item())
            
            # Backward
            optimizer.zero_grad() # clear the gradients of all optimized variables
            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
        
            # Gradient Descent or ADAM step
            optimizer.step()  # update the weights
            
            # Update Progress Bar
            loop.set_description(f"Transformer Trial {trial.number} Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
            # delete intermediate values
            del batch_x, batch_y, out, yTrue, loss
            torch.cuda.empty_cache()
            
        ######################    
        # Validate the model #
        ######################
        model.eval() # prep model for evaluation
        
        with torch.no_grad():
        
            for (idx, batch) in enumerate(val_loader):
                
                batch_x = batch[0].squeeze().to(device)
                batch_y = batch[1].squeeze().to(device)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                out = model(batch_x.float())
                out = out[:,-1,0]
                
                # calculate the loss
                yTrue = batch_y.float() # seq2one
                loss = criterion(out.squeeze(), yTrue)
                
                # record validation loss
                valid_losses.append(loss.item())
                
                # delete intermediate values
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(num_epochs))
        print()
        print_msg = (f'Transformer [{epoch+1:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                     f'train_loss: {train_loss:.5f} ' + 
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)    
        print()
        
        # Update the learning rate
        scheduler.step(valid_loss)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        ## Optuna
        MSE = valid_loss
        trial.report(MSE, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # GPU Memory 
        print("\nGPU Usage after Train/Val Epoch")
        gpu_usage()
        
    ## Save a trained model to a file.
    with open("Transformer_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    return MSE


#%% Train DA-RNN

def train_DARNN(trial, model, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    # train_length = train_loader.dataset.data.shape[1]
    # val_length = val_loader.dataset.data.shape[1]
    
    num_trials = train_loader.dataset.data.shape[1]
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='DARNN_checkpoint.pt')
    
    for epoch in range(num_epochs):
        
        ###################
        # Train the model #
        ###################
        model.train() # prep model for training
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for (idx, batch) in loop: # in enumerate(train_loader)
        
            # NOTE:
                # Create for loop here going through the trials in dim=1
                # This way, the batch can stay as dimensions (timesteps/batch, seq, features) 
                
            # to track training losses across batches and trials
            current_iter_train_loss = []
            
            for t in range(num_trials):
                batch_x = batch[0][:,t].squeeze().to(device)
                batch_y = batch[1][:,t].squeeze().to(device)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                # out = model(batch_x.float(),seqMode)
                out = model(batch_x.float())
                
                # calculate the loss
                yTrue = batch_y.float() # seq2one
                loss = criterion(out.squeeze(), yTrue) # if seq to one, only last timestep ([:,-1]) is used for loss 
                
                # record training loss
                current_iter_train_loss.append(loss.item())
                train_losses.append(loss.item())
                
                # Backward
                optimizer.zero_grad() # clear the gradients of all optimized variables
                loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
            
                # Gradient Descent or ADAM step
                optimizer.step()  # update the weights
                
                # delete intermediate values
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()
            
            # Update Progress Bar
            loop.set_description(f"DA-RNN Trial {trial.number} Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=np.average(current_iter_train_loss))
            
        ######################    
        # Validate the model #
        ######################
        model.eval() # prep model for evaluation
        
        with torch.no_grad():
        
            for (idx, batch) in enumerate(val_loader):
                
                # NOTE:
                    # Same for loop here
                    
                for t in range(num_trials):
                    batch_x = batch[0][:,t].squeeze().to(device)
                    batch_y = batch[1][:,t].squeeze().to(device)
                
                    # batch_x = batch[0].squeeze().to(device)
                    # batch_y = batch[1].squeeze().to(device)
                    
                    # forward pass: compute predicted outputs by passing inputs to the model
                    out = model(batch_x.float())
                    
                    # calculate the loss
                    yTrue = batch_y.float() # seq2seq
                    loss = criterion(out.squeeze(), yTrue)
                    
                    # record validation loss
                    valid_losses.append(loss.item())
                    
                    # delete intermediate values
                    del batch_x, batch_y, out, yTrue, loss
                    torch.cuda.empty_cache()
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(num_epochs))
        print()
        print_msg = (f'DA-RNN [{epoch+1:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                     f'train_loss: {train_loss:.5f} ' + 
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)    
        print()
        
        # Update the learning rate
        scheduler.step(valid_loss)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        ## Optuna
        MSE = valid_loss
        print('\n')
        trial.report(MSE, epoch)
        print('\n')
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # GPU Memory 
        print("\nGPU Usage after Train/Val Epoch")
        gpu_usage()
    
    ## Save a trained model to a file.
    with open("DA-RNN_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    return MSE