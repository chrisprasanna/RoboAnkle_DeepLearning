#%% Imports

from DL_classes import EarlyStopping

import torch
import numpy as np

import optuna
import pickle as pickle # import pickle5 as pickle

from tqdm import tqdm

from GPUtil import showUtilization as gpu_usage


#%% Function
def train_PAFP(trial, model_name, model, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    
    train_length = train_loader.dataset.data.shape[0]
    val_length = val_loader.dataset.data.shape[0]
    
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
    path = f"{model_name}_checkpoint.pt"
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    
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
            
            if model_name == 'GRU':
                # Init hidden state
                batch_size = batch[0].shape[0]
                h = model.init_hidden(batch_size = batch_size)
            
            for t in range(num_trials):
                batch_x = batch[0][:,t].squeeze().to(device)
                batch_y = batch[1][:,t].squeeze().to(device)
                
                # forward pass: compute predicted outputs by passing inputs to the model
                # out = model(batch_x.float(),seqMode)
                if model_name == 'GRU':
                    out,h = model(batch_x.float(),h)
                    out = out[:,-1,0]
                    h = h.detach() # detach hidden in between batches
                else:
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
            
                # Update Progress Bar
                loop.set_description(f"{model_name} Trial {trial.number} Epoch [{epoch + 1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item(), trial=t+1)
                # loop.set_postfix(trial=t+1)
                
                # delete intermediate values
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()
            
        ######################    
        # Validate the model #
        ######################
        model.eval() # prep model for evaluation
        
        with torch.no_grad():
        
            for (idx, batch) in enumerate(val_loader):
                
                # NOTE:
                    # Same for loop here
                    
                if model_name == 'GRU':
                    # Init hidden state
                    batch_size = batch[0].shape[1]
                    h_val = model.init_hidden(batch_size = batch_size)
                    # h_val = model.init_hidden(batch_size = val_length)
                else: 
                    h_val = []
            
                batch_x = batch[0].squeeze().to(device)
                batch_y = batch[1].squeeze().to(device)
                
                # forward pass: compute predicted outputs by passing inputs to the model   
                if model_name == 'GRU':
                    out,h_val = model(batch_x.float(),h_val)
                    out = out[:,-1,0]
                    h_val = h_val.detach() # detach hidden in between batches
                else:
                    out = model(batch_x.float())
                
                # calculate the loss
                yTrue = batch_y.float() # seq2seq
                loss = criterion(out.squeeze(), yTrue)
                
                # record validation loss
                valid_losses.append(loss.item())
                
                # delete intermediate values
                del batch_x, batch_y, out, yTrue, loss
                torch.cuda.empty_cache()    
                    
                # for t in range(num_trials):
                #     batch_x = batch[0][:,t].squeeze().to(device)
                #     batch_y = batch[1][:,t].squeeze().to(device)
                
                #     # batch_x = batch[0].squeeze().to(device)
                #     # batch_y = batch[1].squeeze().to(device)
                    
                #     # forward pass: compute predicted outputs by passing inputs to the model   
                #     if model_name == 'GRU':
                #         out,h_val = model(batch_x.float(),h_val)
                #         out = out[:,-1,0]
                #         h_val = h_val.detach() # detach hidden in between batches
                #     else:
                #         out = model(batch_x.float())
                    
                #     # calculate the loss
                #     yTrue = batch_y.float() # seq2seq
                #     loss = criterion(out.squeeze(), yTrue)
                    
                #     # record validation loss
                #     valid_losses.append(loss.item())
                    
                #     # delete intermediate values
                #     del batch_x, batch_y, out, yTrue, loss
                #     torch.cuda.empty_cache()
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(num_epochs))
        print()
        print_msg = (f'{model_name} Trial {trial.number} Epoch [{epoch+1}/{num_epochs}] ' + 
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
        print("GPU Usage after Train/Val Epoch")
        gpu_usage()
        print('\n')
    
    ## Save a trained model to a file.
    with open("{}_{}.pickle".format(model_name,trial.number), "wb") as fout:
        pickle.dump(model, fout)
    
    return MSE
