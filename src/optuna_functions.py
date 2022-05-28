"""

The purpose of these scripts is to create an Optuna project to optimize
the hyperparameters for several neural network models. These functions are used
in the main train_test_ann_pipeline.py to develop neural network models to
tune controller gains for COBRA in a simulation environment.

"""
# %% Imports

import optuna
from src import training_and_validation
import pickle as pickle
import os

# %% Set Hyperparameters


def set_hyperparameters(trial, hyperparameter_ranges, model_type):
    """
    This function defines and returns hyperparameters, e.g. number of hidden
    layers, for each evaluation of the Optuna objective function.

    Parameters
    ----------
    trial : Optuna Object
        The trial object internally keeps track of which hyperparameters have
        been tried on previous function evaluations.
    hyperparameter_ranges : dictionary
        This is a dictionary that maps hyperparameter names as strings to 
        lists of minimum and maximum allowable values for that hyperparameter.
    model_type : string
        Type of model currently being trained. Currently available options are
        'FFN', 'GRU', and 'DA-GRU'.

    Returns
    -------
    hyperparameters : dict
        Keys are hyperparameter names as strings, values are the scalar value
        Optuna has chosen to use for this trial.

    """

    # unpack all hyperparameter ranges as lists
    sequence_length_ranges = hyperparameter_ranges['sequence length']
    learning_rate_ranges = hyperparameter_ranges['learning rate']
    weight_decay_ranges = hyperparameter_ranges['weight decay']
    gamma_ranges = hyperparameter_ranges['scheduler factor']
    num_layers_ranges = hyperparameter_ranges['number of layers']
    hidden_units_power_ranges = hyperparameter_ranges['hidden units power']
    dropout_ranges = hyperparameter_ranges['dropout factor']
    decoder_hidden_ranges = hyperparameter_ranges['decoder hidden units power']
    batch_size_power_ranges = hyperparameter_ranges['batch size power']
    noise_STD_ranges = hyperparameter_ranges['noise STD']

    # Suggest hyperparameters that are required regardless of which type of
    # network we are currently training.
    sequence_length = trial.suggest_int("sequence length",
                                        sequence_length_ranges[0],
                                        sequence_length_ranges[1])

    learning_rate = trial.suggest_float("learning rate",
                                        learning_rate_ranges[0],
                                        learning_rate_ranges[1])

    weight_decay = trial.suggest_float("weight decay",
                                       weight_decay_ranges[0],
                                       weight_decay_ranges[1])
    
    gamma = trial.suggest_float("gamma", 
                                gamma_ranges[0],
                                gamma_ranges[1])

    batch_size_power = trial.suggest_int('batch size power',
                                         batch_size_power_ranges[0],
                                         batch_size_power_ranges[1])
    batch_size = 2**batch_size_power

    hidden_units_power = trial.suggest_int("hidden units power",
                                           hidden_units_power_ranges[0],
                                           hidden_units_power_ranges[1])
    hidden_size = 2**hidden_units_power

    noise_STD = trial.suggest_float('noise STD',
                                    noise_STD_ranges[0],
                                    noise_STD_ranges[1])

    # Create hyperparameters dictionary to be returned
    hyperparameters = {'sequence length': sequence_length,
                       'learning rate': learning_rate,
                       'weight decay': weight_decay,
                       'scheduler factor': gamma,
                       'hidden size': hidden_size,
                       'batch size': batch_size,
                       'noise STD': noise_STD}

    # Add model-specific hyperparameters
    if (model_type == 'FFN') or (model_type == 'GRU'):

        # Define hyperparameters for the simple feed-forward network or the
        # gated recurrent network, as they have the same hyperparameters

        # number of layers in the network
        num_layers = trial.suggest_int("number of layers",
                                       num_layers_ranges[0],
                                       num_layers_ranges[1])

        # dropout factor
        dropout = trial.suggest_float("dropout",
                                      dropout_ranges[0],
                                      dropout_ranges[1])

        # add hyperparameters to dictionary
        hyperparameters['number of layers'] = num_layers
        hyperparameters['dropout'] = dropout

    elif model_type == 'DA-GRU':

        # Define hyperparameters for the dual attention recurrent network

        decoder_power = trial.suggest_int("decoder size power",
                                          decoder_hidden_ranges[0],
                                          decoder_hidden_ranges[1])
        decoder_size = 2**decoder_power

        # add hyperparameters to dictionary
        hyperparameters['decoder size'] = decoder_size

    else:
        print('The model type is not available.')
        hyperparameters = {}

    return hyperparameters

# %% Report Optuna Objective Function Value Function

def optuna_objective_report(trial, model_type, epoch, number_of_epochs, training_objective_value, validation_objective_value):
    """
    This function reports the objective function value to the optuna study 
    for the current epoch and prints the objective function results. 

    Parameters
    ----------
    trial : optuna module that contains classes and functions
        A Trial instance represents a process of evaluating an objective function. 
        This instance is passed to an objective function and provides interfaces 
        to get parameter suggestion, manage the trial’s state, and set/get 
        user-defined attributes of the trial, so that Optuna users can define a 
        custom objective function through the interfaces.
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    epoch : int
        Current epoch in the training loop
    number_of_epochs : int
        Maximum number of epochs for the training loop.
    training_objective_value : float
        The average achived neural network training loss value for one epoch.
    validation_objective_value : float
        The average achived neural network validation loss value for one epoch.

    Raises
    ------
    optuna
        Optuna is an automatic hyperparameter optimization software framework, 
        particularly designed for machine learning.

    Returns
    -------
    None.

    """
    
    trial.report(validation_objective_value, epoch)
    print_msg = (f'\n{model_type} Trial {trial.number+1} Epoch [{epoch+1}/{number_of_epochs}] ' + 
                 f'train_loss: {training_objective_value:.5f} ' + 
                 f'valid_loss: {validation_objective_value:.5f}\n')
    print(print_msg)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return

# %% Define Optuna Objective Function


def objective(trial, model_type, train_set, val_set, constants, hyperparameter_ranges):
    """
    This function defines the objective function used for automatic hyperparameter
    optimization. For each execution of the objective function (i.e., trial), 
    the hyperparameters are set within the given ranges by the optimizer. Next, 
    the model is trained for the trial and this function returns the final 
    validation loss (i.e., MSE). 

    Parameters
    ----------
    trial : optuna module that contains classes and functions
        A Trial instance represents a process of evaluating an objective function. 
        This instance is passed to an objective function and provides interfaces 
        to get parameter suggestion, manage the trial’s state, and set/get 
        user-defined attributes of the trial, so that Optuna users can define a 
        custom objective function through the interfaces.
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
        This dict contains constant information to be passed across 
        various DNN training and evaluation functions. The data within this 
        dict is not meant to be changed after initialization (i.e., these are 
        values that Optuna does not optimize) and are specific to the modeling
        task (e.g., input size, output size, computing device, etc.).
    hyperparameter_ranges : dict
        This dictionary has the name of each hyperparameter as keys and the
        range that Optuna will search as values. Each input parameter for this 
        function is a list of 2 elements where the first element is the lower 
        bound of the search range and the second element is the upper bound of 
        the search range. Note that some hyperparameters are only used for 
        specific DNN types.

    Returns
    -------
    MSE : float
        The validation loss, the mean squared error, for the trial's best
        performing model. This value is used to evaluate the optuna study and 
        help determine the next trial's hyperparameters. The optuna study
        adapts to minimize this value across trials. 

    """

    # Set hyperparameters for this optimization trial
    hyperparameters = set_hyperparameters(trial, hyperparameter_ranges,
                                          model_type)
    
    # NOTE: It would be great if we had a one-liner below this point that could
    # return MSE and an optimized model. Inputs would be model type, 
    # hyperparameters, constants, and training_set.
    # Something like, "MSE, model = network_functions.train_ann(...)"
    # I think this is the right abstraction level for this function. There
    # would be basically two lines that say 1) define hyperparameters
    # and 2) use hyperparameters to train a model. That's it.
    
    # Train this trial's model
    optuna_flag = True
    MSE = training_and_validation.main_train(model_type, train_set, val_set, constants, hyperparameters, optuna_flag, trial)
    
    return MSE

# %% Build and Optimize Models Using Optuna Study


def optimize_hyperparameters(model_type, train_set, val_set, constants, hyperparameter_ranges):
    """
    This function defines the optuna study, an optimization for a particular 
    model type based on an objective function and user-defined set of constants
    and hyperparameter ranges. Once optimzed, the function retrives the best 
    performing trial and its model using information from the completed study
    and saved pickle files (one for each trial / execution of the objective function). 

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
        This dict contains constant information to be passed across 
        various DNN training and evaluation functions. The data within this 
        dict is not meant to be changed after initialization (i.e., these are 
        values that Optuna does not optimize) and are specific to the modeling
        task (e.g., input size, output size, computing device, etc.).
    hyperparameter_ranges : dict
        This dictionary has the name of each hyperparameter as keys and the
        range that Optuna will search as values. Each input parameter for this 
        function is a list of 2 elements where the first element is the lower 
        bound of the search range and the second element is the upper bound of 
        the search range. Note that some hyperparameters are only used for 
        specific DNN types.

    Returns
    -------
    optimized_model : custom neural network class
        The trained model from the best-performing optuna trial in the study
    best_trial : optuna.study.Study.best_trial
        From an attribute in optuna that returns the best trial in the study

    """

    # Unpack constants
    optuna_trials = constants['number of optuna trials']
    optuna_timeout = constants['optuna timeout']

    # Create an optimization project
    study = optuna.create_study(study_name="Optimize_{}".format(model_type),
                                direction="minimize")

    # Optimize the network
    study.optimize(lambda trial: objective(trial, model_type, train_set, val_set, 
                                           constants, hyperparameter_ranges),                                            
                   n_trials=optuna_trials, timeout=optuna_timeout,
                   gc_after_trial=True)

    # Retrieve best performing trial
    best_trial = study.best_trial

    # Load the best model from pickle file
    project_directory = os.path.dirname(os.getcwd())
    results_directory = os.path.join(project_directory, 'results')  
    pickle_filename = f"{model_type}_{study.best_trial.number}.pickle"
    pickle_file = os.path.join(results_directory, pickle_filename)
    with open(pickle_file, "rb") as fin:
        optimized_model = pickle.load(fin)

    return optimized_model, best_trial
