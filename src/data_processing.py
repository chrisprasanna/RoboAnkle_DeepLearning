# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:54:21 2022

@author: Chris Prasanna

This module will contain functions for sub-sampling data, adding noise to data,
and splitting data into train/validate/test sets.

"""

# %% Imports

import random
from scipy import signal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# %%

def train_val_test_split_data(data_structure, split_percentages=[0.70,0.15,0.15]):
    """
    This function splits the data set. Use examples include splitting data 
    that will be used for model training/validation and data that will be used 
    for model performance evaluation. 

    Parameters
    ----------
    data_structure : dict
        This is the primary data structure in the run_ann_train_test_pipeline.
        Keys are 'file names', 'data', and 'metadata'. 'file names' maps to 
        a list of files that have been imported. 'data' maps to another dict
        where file names map to pandas dataframes full of experimental data.
    split_percentages: array
        3-element array defining the train-validation-test ratio, i.e., the 
        percentage of the total dataset to be dedicated to each of the three
        sub datasets. 

    Returns
    -------
    train_set : dict of pandas dataframes
        Contains all files used for the training dataset, the sample of data 
        used to fit the model. File names map to pandas dataframes. 
    val_set : dict of pandas dataframes
        Contains all files used for the validation dataset, used to provide an 
        unbiased evaluation of a model fit on the training dataset while tuning 
        model hyperparameters. File names map to pandas dataframes. 
    test_set : dict of pandas dataframes
        Contains all files used for the testing dataset, used only to assess 
        the performance of a fully trained model. File names map to pandas dataframes.

    """
    
    # extract file names from data structure and shuffle them into a random
    # order
    file_names = data_structure['file names']
    random.shuffle(file_names)
    
    # compute how many files make up training percentage of the data
    train_set_file_num = int(split_percentages[0]*len(file_names))
    
    # compute how many files make up testing percentage of the data
    test_set_file_num = int(split_percentages[2]*len(file_names))
    
    # compute how many files make up validation percentage of the data
    val_set_file_num = len(file_names) - train_set_file_num - test_set_file_num
    
    # chop out the first N trials for training and the rest for testing
    train_set_names = file_names[0:train_set_file_num]
    val_set_names = file_names[train_set_file_num:train_set_file_num+val_set_file_num]
    test_set_names = file_names[train_set_file_num+val_set_file_num:]
    
    # create an empty dictionary and fill it with dataframes for training
    train_set = {}
    for name in train_set_names:
        train_set[name] = data_structure['data'][name]
        
    # create an empty dictionary and fill it with dataframes for  validation
    val_set = {}
    for name in val_set_names:
        val_set[name] = data_structure['data'][name]
    
    # create an empty dictionary and fill it with dataframes for testing
    test_set = {}
    for name in test_set_names:
        test_set[name] = data_structure['data'][name]

    return train_set, val_set, test_set

#%% 

def resample_data(data_structure, constants):
    """
    This function defines the protocol to resample the data series. The data is 
    resampled to reduce training time and all signals are assumed to be periodic
    because the Fourier method is used. 
    This function is general across all data samples so the data_set variable 
    can be for training, validation, or testing. 

    Parameters
    ----------
    data_structure : dict of pandas dataframes
        The data to be resampled. File names map to pandas dataframes. 
    constants : dict
        This dictionary has the name of each constant as keys and their values. 

    Returns
    -------
    resampled_data : dict of pandas dataframes
        Re-sampled data where filenames map to pandas dataframes. The filenames 
        are consistent with the data_structure input.

    """
    
    # Unpack constants
    fsNew = constants['sub-sample freq']
    fsOriginal = constants['original sample freq']
    num_timestepsPerTrial = constants['number of timesteps per trial']
    
    fsRatio = fsNew / fsOriginal
    num_timestepsPerTrial = int(num_timestepsPerTrial * fsRatio)
    
    # Pre-allocate
    resampled_data = data_structure.copy()
    
    # Re-sample data
    for file in data_structure:
        trial = data_structure[file]
        resampled_signals = signal.resample(trial, num_timestepsPerTrial, axis=0)
        resampled_data[file] = pd.DataFrame(resampled_signals, columns=trial.columns)
        
    # Update the number of timesteps constant
    constants['resampled number of timesteps per trial'] = len(resampled_signals)
    
    return resampled_data

#%% 

def get_lookback_windows(data_structure, sequence_length, constants):
    """
    This function defines the protocol that splits a data structure into feature
    and target signals while also defining lookback windows for the feature signals.
    A prediction horizon can also be included to train models that predict multiple 
    samples into the future. Note that the column names of the pandas dataframes
    in the data_structure input are specific to the COBRA project. If you use 
    this code for a different project, these column names will need to be edited. 

    Parameters
    ----------
    data_structure : dict of pandas dataframes
        Data to be split into features/targets and have lookback periods defined
    sequence_length : int
        Sequence Length is the length of the historical sequence of input data. 
        If the sequence length is N, then the window range of the input data 
        will be from samples [0:-1,-2,...,-N].
    constants : dict
        This dictionary has the name of each constant as keys and their values.

    Returns
    -------
    features : numpy array
        Model input feature data. Dimensions represent 
        [walking trial number, timestep, sequence/lookback, feature signal]
    targets : numpy array
        Model output/training target data. Dimensions represent 
        [walking trial number, timestep, target signal]

    """
    
    # Unpack constants 
    input_size = constants['input size']
    output_size = constants['output size']
    num_timestepsPerTrial = constants['resampled number of timesteps per trial']
    prediction_horizon = constants['prediction horizon']
    
    # Get number of walking trials for this dataset
    num_trials = len(data_structure)
    
    # Pre-allocate
    features = np.zeros((num_trials, num_timestepsPerTrial, sequence_length, input_size))
    targets = np.zeros((num_trials, num_timestepsPerTrial, output_size))
    
    for n, file in enumerate(data_structure):
        
        # Retrieve Output
        ankle_torque = data_structure[file]['PAFP Torque (Nm/kg)']
        
        # Retrieve States
        omega_motor = data_structure[file]['Motor Velocity (rpm)']
        hip_position = data_structure[file]['Hip Angle (rad)']
        ankle_position = data_structure[file]['Ankle Angle (rad)']
        left_force = data_structure[file]['Left vGRF (N)']
        right_force = data_structure[file]['Right vGRF (N)']
        
        # Retrieve Input
        U = data_structure[file]['Motor Current Command (A)']
        
        tmp_x = np.array([omega_motor, hip_position, ankle_position, left_force, right_force, U])
        data_x = pd.DataFrame(np.transpose(tmp_x),
                           columns=['w_m','th_h','th_a','F_l','F_r','i_m'])
        tmp_y = np.array([ankle_torque])
        data_y = pd.DataFrame(np.transpose(tmp_y),
                           columns=['Tau'])
        
        for i, signal_name in enumerate(list(data_x.columns)):
            for j in range(sequence_length):
                features[n, :, j, i] = data_x[signal_name].shift(sequence_length - j - 1).fillna(method="bfill")
        
                
        for i, signal_name in enumerate(list(data_y.columns)):
            targets[n,:,i] = data_y[signal_name].shift(-prediction_horizon).fillna(method="ffill").values
    
        
    features = features[:,sequence_length:-1]
    targets = targets[:,sequence_length:-1]
    
    return features, targets

#%% 

def normalize_data(features, targets, constants, is_training_set):
    """
    This function defines the data scaling protocol that normalizes each signal
    (input and target) based off their maximum and minimum values found within
    the training dataset. All signals are scaled to [0,1]. This definition 
    includes functionality for both training and non-training datasets. Note that
    target signals are normalized since the neural networks predict multiple output
    signals (i.e., MIMO). 

    Parameters
    ----------
    features : numpy array
        Model input feature data. Dimensions represent 
        [walking trial number, timestep, sequence/lookback, feature signal]
    targets : numpy array
        Model output/training target data. Dimensions represent 
        [walking trial number, timestep, target signal]
    constants : dict
        This dictionary has the name of each constant as keys and their values.
    is_training_set : bool
        Boolean where a value of 1 indicates that the data_set input is a training 
        dataset and a value of 0 inficates that the data_set input is not a 
        training dataset

    Returns
    -------
    normalized_features : numpy array
        Model input feature data normalized from [0,1]. Array dimensions are kept
        consistent with the features input. 
    normalized_targets : numpy array
        Model output/training target data normalized from [0,1]. Array dimensions 
        are kept consistent with the targets input. 

    """
    
    if is_training_set:
        features_max = features.max(axis=(0,1,2))
        features_min = features.min(axis=(0,1,2))
        targets_max = targets.max(axis=(0,1))
        targets_min = targets.min(axis=(0,1))        
    else:
        features_max = constants['Maximum Features']
        features_min = constants['Minimum Features']
        targets_max = constants['Maximum Targets']
        targets_min = constants['Minimum Targets']
    
    normalized_features = (features - features_min) / (features_max - features_min)
    normalized_targets = (targets - targets_min) / (targets_max - targets_min) 
    
    # Update constants
    constants['Maximum Features'] = features_max
    constants['Minimum Features'] = features_min
    constants['Maximum Targets'] = targets_max
    constants['Minimum Targets'] = targets_min
    
    return normalized_features, normalized_targets

#%%

def convert_to_tensors(features, targets):
    """
    This function converts a numpy ndarray to a pytorch tensor so training can
    run on GPUs.  Additionally, tensors are used to encode the inputs and outputs 
    of a model, as well as the model’s parameters.

    Parameters
    ----------
    features : numpy array
        Model input feature data. Dimensions represent 
        [walking trial number, timestep, sequence/lookback, feature signal]
    targets : numpy array
        Model output/training target data. Dimensions represent 
        [walking trial number, timestep, target signal]

    Returns
    -------
    tensor_features : pytorch tensor
        Model input feature data converted to a tensor with dimensions
        consistent with the features input numpy array. 
    tensor_targets : pytorch tensor
        Model output target data converted to a tensor with dimensions
        consistent with the targets input numpy array.

    """
    
    tensor_features = torch.Tensor(features)
    tensor_targets = torch.Tensor(targets)
    
    return tensor_features, tensor_targets

#%%

def add_guassian_input_noise(features, noiseSTD):
    """
    This function defines the method to add Gaussian noise to the input feature 
    data which reduces the likelihood of overfitting. Additive noise should be 
    included for training and validation datasets but excluded from test datasets. 
    In addition, noise should not be added to output target data. 

    Parameters
    ----------
    features : pytorch tensor
        Model input feature data. Dimensions represent 
        [walking trial number, timestep, sequence/lookback, feature signal]
    noiseSTD : float
        The standard deviation of the noise injected on to the training data. 
        This random noise is added to each training sample to reduce the chance
        of overfitting.

    Returns
    -------
    noisy_features : pytorch tensor
        Model input feature data with additive gaussian noise. Tensor dimensions 
        are kept consistent with the features input.  

    """
    
    noisy_features = features  + noiseSTD*torch.randn(features.size())
    
    return noisy_features

#%% 

def create_torch_datasets(features, targets, train_bool):
    """
    This function defines the conversion from feature and target tensors to a 
    custom tensor dataset used for learning. The CustomTensorDataset class 
    implements __init__ and __getitem__ functions. The __init__ function is run 
    once when instantiating the Dataset object. We initialize the directory 
    containing feature and target signals. The __getitem__ function loads 
    and returns a sample from the dataset at the given index. 

    Parameters
    ----------
    features : pytorch tensor
        Model input feature data. Dimensions represent 
        [walking trial number, timestep, sequence/lookback, feature signal]
    targets : pytorch tensor
        Model output/training target data. Dimensions represent 
        [walking trial number, timestep, target signal]
    train_bool : bool
        Variable specifying if the data is for a training set (1) or not (0). 
        This must be specified since training data is permuted so that batches
        occur for the dimensions representing timesteps. 

    Returns
    -------
    torch_data_set : tensor dataset
        Custom torch dataset
    """
    
    ## Dataset Class
    class CustomTensorDataset(TensorDataset):
        """TensorDataset with support of transforms.
        """
        def __init__(self, tensors):
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
            self.tensors = tensors
            self.data = tensors[0]
            self.targets = tensors[1]


        def __getitem__(self, index):
            x = self.tensors[0][index]

            y = self.tensors[1][index]

            return x, y
    
    ## Create Data Loaders
    if train_bool:
        
        # (timesteps, trials, sequence, features)
        # This way we are dividing batches by timesteps and not trials
        features = features.permute(1,0,2,3)
        targets = targets.permute(1,0,2)
        
        # Create dataset
        torch_data_set = CustomTensorDataset([features, targets])
        
    else:
        
        # Create Dataset (trials, timesteps, sequence, features)
        torch_data_set = CustomTensorDataset([features, targets])
    
    
    return torch_data_set

#%%

def create_dataloaders(data_set, hyperparameters, is_training_set):
    """
    This function defines the creation of custom pytorch dataloaders using the 
    custom pytorch datasets. Dataloders help us to load data on GPUs and iterate 
    over elements in a dataset. This class also enables batching and multi-processing.
    If the dataset is specified as a training dataset, then batching is enabled. 
    Otherwise, the batch size is set to 1, which is appropriate for model evaluation. 

    Parameters
    ----------
    data_set : tensor dataset
        Custom torch dataset that includes feature data and target data
    hyperparameters : dict
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
    
    # Retrieve batch size
    if is_training_set:
        batch_size = hyperparameters['batch size']
    else:
        batch_size = 1
    
    # Create Dataloaders
    torch_data_loader = torch.utils.data.DataLoader(data_set,batch_size=batch_size,
                                               shuffle=False, drop_last=True)
    
    return torch_data_loader