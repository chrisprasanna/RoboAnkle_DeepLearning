"""
Created on Wed Feb  9 10:01:23 2022

@author: Chris Prasanna

This file contains functions related to initializing the constants and 
hyperparameters that are used throughout DNN training and evaluation scripts. 
The primary purpose of all of these functions is to return a dict structure to 
our primary analysis scripts.
"""

import torch.nn as nn

# %% Get Constants


def get_constants(device='cpu',
                  PATH="../results/training results",
                  input_size=1,
                  output_size=1,
                  lr_step_size=1,
                  lr_patience=1,
                  min_lr=1e-5,
                  num_trials=1,
                  num_timestepsPerTrial=10000,
                  criterion=nn.MSELoss(),
                  num_epochs=1000,
                  optuna_trials=3,
                  optuna_timeout=None,
                  prediction_horizon=1,
                  fs=1000,
                  fsNew=30
                  ):
    """
    Set constants (parameters not optimized with Optuna) and return as a 
    dictionary structure

    Parameters
    ----------
    device : string
        An object that represents the device on which a tensor will be 
        allocated during the DNN training process. 
        The default is 'cpu'.
    PATH : string
        The filepath that is used to save the DNN evaluation results. 
        The default is a results folder within the project.
    input_size : int
        The number of DNN input features used as predictors. 
        The default is 1.
    output_size : int
        The number of DNN output target variables (response). 
        The default is 1.
    lr_step_size : int
        Number of epochs with no improvement after which learning rate will be 
        reduced. For example, if lr_step_size = 2, then we will ignore the 
        first 2 epochs with no improvement, and will only decrease the LR after 
        the 3rd epoch if the loss still hasn’t improved then. 
        The default is 1.
    lr_patience : int
        Number of epochs to wait before resuming normal operation after lr has 
        been reduced.
        The default is 1.
    min_lr : float
        A lower bound on the learning rate. 
        The default is 1e-5.
    num_trials : int
        The number of data trials within the dataset. This value is important 
        for constructing tensors and training batches. 
        The default is 1.
    num_timestepsPerTrial : int
        The number of timesteps (samples) per data trial.
        The default is 10000.
    criterion : torch.nn.modules.loss
        The loss function used during the DNN training process. Loss functions 
        are used to gauge the error between the prediction output and the 
        provided target value.
        The default is torch.nn.MSELoss().
    num_epochs : int, optional
        An epoch is a measure of the number of times all training data is used 
        once to update the parameters
        The default is 1000.
    optuna_trials : int, optional
        The number of iterations that Optuna executes its evaluation of an 
        objective function. Each trial suggests values of hyperparameters based
        on Optuna's search crtieria. 
        The default is 100.
    optuna_timeout : float, optional
        Stop Optuna study after the given number of second(s). If this argument 
        is set to None, the study is executed without time limitation. 
        The default is None.
    Data : dict, optional
        Data structure from "import_data_structure" function. 
        This dictionary has keys for "data", "metadata", and "file names".
        "file names" links to a list of file names. "data" and "metadata" link
        to additional dictionaries where file names are keys and values are
        Pandas dataframes with data and metadata, respectively.
        The default is a blank dictionary.
    prediction_horizon : int, optional
        How many timesteps/samples ahead the DNN model predicts into the 
        future.
        The default is 1.
    fs : int
        The sample rate that the time series data was collected in Hz.
        The default is 1000.
    fsNew : int
        The value to resample the time series data in Hz. The resampled signal 
        starts at the same value as the original signal but is sampled 
        differently acording to fsNew. A Fourier method is used so the time
        series signals must be periodic. 
        The default is 30.

    Returns
    -------
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
        This dict contains constant information to be passed across 
        various DNN training and evaluation functions. The data within this 
        dict is not meant to be changed after initialization (i.e., these are 
        values that Optuna does not optimize)

    """

    constants = {'device': device,
                 'results path': PATH,
                 'input size': input_size,
                 'output size': output_size,
                 'learning rate scheduler delay': lr_step_size,
                 'learning rate scheduler patience': lr_patience,
                 'minimum learning rate': min_lr,
                 'number of walking trials': num_trials,
                 'number of timesteps per trial': num_timestepsPerTrial,
                 'loss function': criterion,
                 'max number of epochs': num_epochs,
                 'number of optuna trials': optuna_trials,
                 'optuna timeout': optuna_timeout,
                 'prediction horizon': prediction_horizon,
                 'original sample freq': fs,
                 'sub-sample freq': fsNew
                 }

    return constants


# %% Get Hyperparamter Ranges

def get_hyperparameter_ranges(seq_len=[2, 20],
                              lr=[1e-5, 1e-1],
                              weight_decay = [1e-5, 1e-1],               
                              gamma = [0.1, 0.9],
                              num_layers=[1, 3],
                              num_HUs_pow=[4, 9],
                              dropout=[0.1, 0.5],
                              P=[4, 7],
                              batch_pow=[4, 8],
                              noise_std=[0.1, 1]
                              ):
    """
    Sets the hyperparameter ranges used during the Optuna optimization trials. 
    Each input parameter is a list of two elements (lower and upper bounds of 
    the hyperparameter range) and the output is a dictionary containing the 
    names of each hyperparameter and the range values. 

    Parameters
    ----------
    seq_len : list of int, optional
        Sequence Length is the length of the historical sequence of input data. 
        If the sequence length is N, then the window range of the input data 
        will be from samples [0:-1,-2,...,-N].
        The default is [2,20].
    lr : list of float, optional
        The learning rate is a hyperparameter that controls how much to change 
        the model in response to the estimated error each time the model 
        weights are updated.
        The default is [1e-5, 1e-1].
    gamma : list of float, optional
        The weight decay which is a regularization technique used by adding a 
        small penalty (L2 norm of all the weights of the model) to the loss 
        function.
        The default is [1e-5, 1e-1].
    num_layers : list of int, optional
        The number of layers in the DNN
        The default is [1,3].
    num_HUs_pow : list of int, optional
        The powers of 2 for the neural network layers' number of hidden units.
        Searching in powers of two reduces the search space logarithmically.
        The default is [4,9].
    dropout : list of float, optional
        Probability of an element to be zeroed during training. This is a 
        regularization technique used to avoid overfitting. 
        The default is [0.1, 0.5].
    P : list of int, optional
        The powers of 2 for the DARNN's number of decoder units.
        The default is [4,7].
    batch_pow : list of int, optional
        The powers of 2 for the batch size. Batch size is a term used in 
        machine learning and refers to the number of training examples 
        utilized in one iteration.
        The default is [4,8].
    noise_std : list of float, optional
        The standard deviation of the noise injected on to the training data. 
        This random noise is added to each training sample to reduce the chance
        of overfitting.
        The default is [0.1, 1].

    Returns
    -------
    hp_ranges : dict
        This dictionary has the name of each hyperparameter as keys and the
        range that Optuna will search as values. Each input parameter for this 
        function is a list of 2 elements where the first element is the lower 
        bound of the search range and the second element is the upper bound of 
        the search range. Note that some hyperparameters are only used for 
        specific DNN types. 

    """

    hp_ranges = {
        'sequence length': seq_len,
        'learning rate': lr,
        'weight decay': weight_decay,
        'scheduler factor': gamma,
        'number of layers': num_layers,
        'hidden units power': num_HUs_pow,
        'dropout factor': dropout,
        'decoder hidden units power': P,
        'batch size power': batch_pow,
        'noise STD': noise_std
    }

    return hp_ranges
