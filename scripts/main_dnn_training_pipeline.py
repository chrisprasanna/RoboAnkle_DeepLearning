"""

This is the main pipeline for building, training, and testing the DNNs. 

@author: Chris Prasanna

"""

from src import set_parameters
from src import data_processing
from src import optuna_functions
from src import testing_and_evaluation
from src import save_results

import pickle

# %% Load experimental data

with open('../data/data_structure.pkl', 'rb') as f:
    data_structure = pickle.load(f)
del f

# %% Define Constants and Hyperparameters

# This function gets all constants used in the script (metadata, filepaths,
# number of trials, timesteps per trial, etc.)
constants = set_parameters.get_constants(input_size=6, output_size=1, optuna_trials=100, num_timestepsPerTrial=3600, fs=120)

# This function gets all hyperparameter ranges used in the Optuna optimization
hyperparameter_ranges = set_parameters.get_hyperparameter_ranges()

# %% Cut data into train vs test sets

train_set, validation_set, test_set = data_processing.train_val_test_split_data(data_structure, split_percentages=[0.70,0.15,0.15])

# %% Choose and Train Model(s)

model_types = ('FFN', 'GRU', 'DA-GRU')

for model_type in model_types:

    ## Optimize Hyperparameters
    optimized_model, best_trial = optuna_functions.optimize_hyperparameters(
        model_type=model_type, train_set=train_set, val_set=validation_set, constants=constants, hyperparameter_ranges=hyperparameter_ranges)
    
    ## Retrieve sequence length from best neural network
    sequence_length = best_trial.params['sequence length']
    
    ## Compute & visualize test results
    test_results = testing_and_evaluation.main_test(model=optimized_model, model_type=model_type, data_set=test_set, constants=constants, sequence_length=sequence_length)
    
    ## Save Data / Results
    save_results.main_save(model_type=model_type, optimized_model=optimized_model, test_results=test_results)

print("\n\n*** Finished ***")