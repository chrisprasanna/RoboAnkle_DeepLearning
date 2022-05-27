# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:39:02 2022

@author: Anthony Anderson & Chris Prasanna

"""

# %% Imports
from src import define_model_functions
from src import data_processing
from src import optuna_functions

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from GPUtil import showUtilization as gpu_usage
import pickle as pickle
import os

#%%

def preprocess_data_for_training(data_set, hyperparameters, constants, is_training_set):
    """
    This function defines the pipeline that processes the data samples and 
    prepares them for training. Sub-functions represent different data 
    processing procedures and are defined in data_processing.py. 

    Parameters
    ----------
    data_set : dict of pandas dataframes
        Dataset used for this preprocessing function. This function is general
        across all data samples so the data_set variable can be for training, 
        validation, or testing. File names map to pandas dataframes. 
    hyperparameters : dict
        Keys are hyperparameter names as strings, values are the scalar value
        Optuna has chosen to use for this trial.
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
    is_training_set : bool
        Boolean where a value of 1 indicates that the data_set input is a training 
        dataset and a value of 0 inficates that the data_set input is not a 
        training dataset

    Returns
    -------
    torch_data_loader : torch dataloader class
        Processed dataset converted into a PyTorch DataLoader that can iterate 
        through the data during training and evaluation as needed

    """
    
    # Resample data
    # NOTE: this could be used outside Optuna so it's not called repeatedly
    resampled_data = data_processing.resample_data(data_set, constants)
    
    # Build lookback window
    features, targets = data_processing.get_lookback_windows(resampled_data, hyperparameters['sequence length'], constants)
    
    # Data normalization
    normalized_features, normalized_targets = data_processing.normalize_data(features, targets, constants, is_training_set)
    
    # Convert data to tensors
    tensor_features, tensor_targets = data_processing.convert_to_tensors(normalized_features, normalized_targets)
    
    # Add noise to input feature data
    noisy_features = data_processing.add_guassian_input_noise(tensor_features, hyperparameters['noise STD'])
    
    # Create datasets
    torch_data_set = data_processing.create_torch_datasets(noisy_features, tensor_targets, train_bool=1)
    
    # Create loaders
    torch_data_loader = data_processing.create_dataloaders(torch_data_set, hyperparameters, is_training_set)
    
    return torch_data_loader

#%% 

def set_up_trainer(hyperparameters, constants, model, optimizer_name="AdamW"):
    """
    This function defines the optimizer and scheduler torch objects used during
    neural network training. See the TORCH.OPTIM page on pytorch.org for more
    options and details. 

    Parameters
    ----------
    hyperparameters : dict
        Keys are hyperparameter names as strings, values are the scalar value
        Optuna has chosen to use for this trial.
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
    model : torch.nn.Module
        Empty neural network torch class containting custom architecture information
    optimizer_name : string, optional
        Optimization algorithm description. The default is "AdamW".

    Returns
    -------
    optimizer : torch.optim.Optimizer 
        Training optimizer object, which holds the training state and will 
        update the parameters based on the computed gradients.
    scheduler : torch.optim.lr_scheduler
        Object that provides the method to adjust the learning rate based on
        the validation losses over epochs. Learning rate scheduling should be 
        applied after optimizer’s update

    """

    # Unpack Hyperparameters
    learning_rate = hyperparameters['learning rate']
    weight_decay = hyperparameters['weight decay']
    gamma = hyperparameters['scheduler factor']
    
    # Unpack Constants
    lr_step_size = constants['learning rate scheduler delay']
    lr_patience = constants['learning rate scheduler patience']
    lr_min= constants['minimum learning rate']
    
    # Generate the Optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Generate the Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, 
                                                 patience=lr_patience, cooldown=lr_step_size, 
                                                 min_lr=lr_min, verbose=True)

    return optimizer, scheduler

#%%

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=1e-4, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            *** NOTE: default changed to 1e-4
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print('\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
#%% 

def train_neural_network(train_set, model, model_type, optuna_flag, trial, epoch, optimizer, criterion, constants):
    """
    This function defines the neural network training and learning procedure. 

    Parameters
    ----------
    train_set : pytorch dataloader object
        Contains all files used for the training dataset, the sample of data 
        used to fit the model. File names map to pandas dataframes.      
    model : torch.nn.Module
        The neural network torch object that is being trained, which includes
        custom architecture information and model parameters
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    optuna_flag : bool
        This flag needs to be raised if conducting an optuna hyperparameter 
        optimization. This value indicates to the training pipeline that
        validation loss values must be reported to the optuna study. 
    trial : optuna module that contains classes and functions, optional
        A Trial instance represents a process of evaluating an objective function. 
        This instance is passed to an objective function and provides interfaces 
        to get parameter suggestion, manage the trial’s state, and set/get 
        user-defined attributes of the trial, so that Optuna users can define a 
        custom objective function through the interfaces.
    epoch : int
        Current epoch in the training loop
    optimizer : torch.optim.Optimizer 
        Training optimizer object, which holds the training state and will 
        update the parameters based on the computed gradients.
    criterion : torch.nn Loss Function
        A serializable function where, given an input and a target, computes a 
        gradient according to a given loss function. 
    constants : dict
        This dictionary has the name of each constant as keys and their values. 

    Returns
    -------
    train_losses : numpy array
        An array of achived training loss values where the array induces represent
        each training loop iteration
        
    """
    
    # Unpack constants
    device = constants['device']
    number_of_epochs = constants['max number of epochs']
    

    # Pre-allocations
    train_losses = []
    
    # prep model for training
    model.train() 
    
    # Set up TQDM
    loop = tqdm(enumerate(train_set), total=len(train_set), leave=False)
    
    # Loop through batches
    for (idx, batch) in loop:
        # If GRU, initialize hidden state
        if model_type == 'GRU':
            h = model.init_hidden(batch_size = batch[0].shape[0])
        
        # Loop through walking trials
        number_of_walking_trials = np.size(batch[0],axis=1)
        for walking_trial in range(number_of_walking_trials):
            
            # Extract features and targets
            features = batch[0][:,walking_trial].squeeze().to(device)
            targets = batch[1][:,walking_trial].squeeze().to(device)
            
            # Forward pass
            if model_type == 'GRU':
                model_prediction,h = model(features.float(),h)
                model_prediction = model_prediction[:,-1,:] # get last timestep of model prediction
                h = h.detach() # detach hidden in between batches
            else:
                model_prediction = model(features.float())
        
            # Compute loss and record
            loss_object = criterion() # instantiate loss class 
            loss = loss_object(model_prediction, targets.float()) # use object to compute loss
            train_losses.append(loss.item()) # store loss value
            
            # Backwards pass
            optimizer.zero_grad() # clear the gradients of all optimized variables
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
        
            # Optimizer step / update the weights
            optimizer.step()
            
            # Update TQDM progress bar
            if optuna_flag:
                loop.set_description(f"{model_type} Trial {trial.number} Epoch [{epoch + 1}/{number_of_epochs}]")
            else:
                loop.set_description(f"{model_type} Epoch [{epoch + 1}/{number_of_epochs}]")
            loop.set_postfix(loss=loss.item(), trial=walking_trial+1)
            
            # Delete intermediate variables and empty cache
            del features, targets, model_prediction, loss
            torch.cuda.empty_cache()
    
    return train_losses
   
#%% 
def validate_neural_network(model, model_type, val_set, criterion, constants):
    """
    This function defines the neural network validation procedure. 

    Parameters
    ----------
    model : torch.nn.Module
        The neural network torch object that is being trained, which includes
        custom architecture information and model parameters
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    val_set : dict of pandas dataframes
        Contains all files used for the validation dataset, used to provide an 
        unbiased evaluation of a model fit on the training dataset while tuning 
        model hyperparameters. File names map to pandas dataframes.
    criterion : torch.nn Loss Function
        A serializable function where, given an input and a target, computes a 
        gradient according to a given loss function. 
    constants : dict
        This dictionary has the name of each constant as keys and their values. 

    Returns
    -------
    validation_losses : numpy array
        An array of achived validation loss values where the array induces represent
        each training loop iteration

    """
    
    # Unpack constants
    device = constants['device']
    
    # pre-allocate
    validation_losses = []
    
    # prep model for evaluation
    model.eval()
    with torch.no_grad():
        
        # Loop through validation batches
        for (idx, batch) in enumerate(val_set):
            # If GRU, initialize hidden state
            if model_type== 'GRU':
                h_val = model.init_hidden(batch_size = batch[0].shape[1])
            
            # Extract features and targets
            features = batch[0].squeeze().to(device)
            targets = batch[1].squeeze().to(device)
            
            # Forward pass
            if model_type == 'GRU':
                model_prediction,h_val = model(features.float(),h_val)
                model_prediction = model_prediction[:,-1,:]
                h_val = h_val.detach() # detach hidden in between batches
            else:
                model_prediction = model(features.float())
        
            # Compute loss and record
            loss_object = criterion() # instantiate loss class
            loss = loss_object(model_prediction, targets.float()) # use object to compute loss
            validation_losses.append(loss.item())
            
            # Delete intermediate variables and empty cache
            del features, targets, model_prediction, loss
            torch.cuda.empty_cache() 
    
    return validation_losses

#%% 

def neural_network_training_loop(model, train_set, val_set, model_type, optimizer, scheduler, constants, optuna_flag, trial):
    """
    This function defines the neural network training loop and calls other per-Epoch
    activities (e.g., early stopping, learning rate scheduling, optuna reporting).
    This function returns the validation loss from the best-performing neural
    network model. In addition, the best-performing model is saved to a pickle
    file for future use. 

    Parameters
    ----------
    model : torch.nn.Module
        Empty neural network torch class containting custom architecture information
    train_set : dict of pandas dataframes
        Contains all files used for the training dataset, the sample of data 
        used to fit the model. File names map to pandas dataframes. 
    val_set : dict of pandas dataframes
        Contains all files used for the validation dataset, used to provide an 
        unbiased evaluation of a model fit on the training dataset while tuning 
        model hyperparameters. File names map to pandas dataframes.
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    optimizer : torch.optim.Optimizer 
        Training optimizer object, which holds the training state and will 
        update the parameters based on the computed gradients.
    scheduler : torch.optim.lr_scheduler
        Object that provides the method to adjust the learning rate based on
        the validation losses over epochs. Learning rate scheduling should be 
        applied after optimizer’s update
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
    optuna_flag : bool
        This flag needs to be raised if conducting an optuna hyperparameter 
        optimization. This value indicates to the training pipeline that
        validation loss values must be reported to the optuna study. 
    trial : optuna module that contains classes and functions, optional
        A Trial instance represents a process of evaluating an objective function. 
        This instance is passed to an objective function and provides interfaces 
        to get parameter suggestion, manage the trial’s state, and set/get 
        user-defined attributes of the trial, so that Optuna users can define a 
        custom objective function through the interfaces.

    Returns
    -------
    MSE : float
        The validation loss, the mean squared error, for the trial's best
        performing model. This value is used to evaluate the optuna study and 
        help determine the next trial's hyperparameters. The optuna study
        adapts to minimize this value across trials.

    """
    
    # Unpack constants
    number_of_epochs = constants['max number of epochs']
    
    # Pre-allocate
    validation_MSE_values = []
    
    # Define criterion / cost function
    criterion = torch.nn.MSELoss
    
    # Set up file paths for saving results and checkpoints
    project_directory = os.path.dirname(os.getcwd())
    results_directory = os.path.join(project_directory, 'results')    
    
    # Set up early stopping pipeline
    early_stopping_patience = 10
    checkpoint_filepath = os.path.join(results_directory, f"{model_type}_checkpoint.pt")
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint_filepath)
    
    # ============================ #
    # Training and Validation Loop #
    # ============================ #
    for epoch in range(number_of_epochs):
        
        # Train the model
        train_losses = train_neural_network(train_set, model, model_type, optuna_flag, trial, epoch, optimizer, criterion, constants)
        training_MSE = np.mean(train_losses)
          
        # Validate the model
        validation_losses = validate_neural_network(model, model_type, val_set, criterion, constants)
        validation_MSE = np.mean(validation_losses)
        validation_MSE_values.append(validation_MSE)
        
        # Update learning rate via scheduler
        scheduler.step(validation_MSE)
        
        # Report to Optuna and print training results
        if optuna_flag:
            optuna_functions.optuna_objective_report(trial, model_type, epoch, number_of_epochs, training_MSE, validation_MSE)
        else:
            print(f'\n{model_type} Epoch [{epoch+1}/{number_of_epochs}] ' + 
                         f'train_loss: {training_MSE:.5f} ' + 
                         f'valid_loss: {validation_MSE:.5f}\n') 
        
        # Report to early stopping pipeline
        early_stopping(validation_MSE, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # Print GPU memory usage
        gpu_usage()
    
    
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_filepath))
    
    # Save trained model to a pickle file
    if optuna_flag:
        pickle_filename = "{}_{}.pickle".format(model_type,trial.number)
    else:
        pickle_filename = "{}.pickle".format(model_type)
    
    pickle_file = os.path.join(results_directory, pickle_filename)         
    with open(pickle_file, "wb") as fout:
        pickle.dump(model, fout)
    
    # Return minimum validation loss
    MSE = np.min(np.array(validation_MSE_values))
    return MSE

# %%

def main_train(model_type, train_set, val_set, constants, hyperparameters, optuna_flag, trial=None):
    """
    This function defines the main neural network training protocol. First, the 
    model architecture is defined. Next, the data is processed and prepared for 
    training. Next, the training objects are defined. Finally, the model is 
    trained and validated. The function returns the validation loss of the final
    trained model. 

    Parameters
    ----------
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    train_set : dict of pandas dataframes
        Contains all files used for the training dataset, the sample of data 
        used to fit the model. File names map to pandas dataframes. 
    val_set : dict of pandas dataframes
        Contains all files used for the validation dataset, used to provide an 
        unbiased evaluation of a model fit on the training dataset while tuning 
        model hyperparameters. File names map to pandas dataframes.  
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
    hyperparameter_ranges : dict
        This dictionary has the name of each hyperparameter as keys and the
        range that Optuna will search as values. Each input parameter for this 
        function is a list of 2 elements where the first element is the lower 
        bound of the search range and the second element is the upper bound of 
        the search range.
    optuna_flag : bool
        This flag needs to be raised if conducting an optuna hyperparameter 
        optimization. This value indicates to the training pipeline that
        validation loss values must be reported to the optuna study. 
    trial : optuna module that contains classes and functions, optional
        A Trial instance represents a process of evaluating an objective function. 
        This instance is passed to an objective function and provides interfaces 
        to get parameter suggestion, manage the trial’s state, and set/get 
        user-defined attributes of the trial, so that Optuna users can define a 
        custom objective function through the interfaces.

    Returns
    -------
    MSE : float
        The validation loss, the mean squared error, for the trial's best
        performing model. This value is used to evaluate the optuna study and 
        help determine the next trial's hyperparameters. The optuna study
        adapts to minimize this value across trials. 

    """
    
    # Build model
    model = define_model_functions.get_neural_network(model_type,
                                                      hyperparameters,
                                                      constants)
    
    # Preprocess training and validation data
    processed_train_set = preprocess_data_for_training(train_set, hyperparameters, constants, is_training_set=True)
    processed_val_set = preprocess_data_for_training(val_set, hyperparameters, constants, is_training_set=False)
    
    # Set up trainer
    optimizer, scheduler = set_up_trainer(hyperparameters, constants, model)
    
    # Train and validate model
    MSE = neural_network_training_loop(model, processed_train_set, processed_val_set, model_type, optimizer, scheduler, constants, optuna_flag, trial)
    
    return MSE
                                    