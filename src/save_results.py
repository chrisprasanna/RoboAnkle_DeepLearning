"""
Created on Fri Apr 29 10:34:14 2022

@author: Chris Prasanna
"""

import os
import torch
import pickle as pickle

# %%

def save_neural_network(model_type, optimized_model, results_directory): 
    """
    This function is responsible for saving the full-trained neural network
    after the hyperparameter optimization protocol. 

    Parameters
    ----------
    model_type : string
        Type of model currently being trained. Currently available options are
        'FFN', 'GRU', and 'DA-GRU'.
    optimized_model : custom neural network class
        The trained model from the best-performing optuna trial in the study
    results_directory : string
        Local directory to save the testing results to

    Returns
    -------
    None.

    """
    # Saved file path 
    trained_model_filepath = os.path.join(results_directory, f'{model_type}_trained_model.pt')
    
    # If GRU, set hidden state batch size to 1
    if model_type == 'GRU':
        optimized_model.init_hidden(batch_size = 1)
    
    # Save learnable model parameters
    model_parameters = optimized_model.state_dict()
    torch.save(model_parameters, trained_model_filepath)
    
    return

# %%
def main_save(model_type, optimized_model, test_results):
    """
    This is the main function responsible for saving all the results from the 
    neural network testing protocol. In addition, it calls a function to save
    the fully-trained neural network model. 

    Parameters
    ----------
    model_type : string
        Type of model currently being trained. Currently available options are
        'FFN', 'GRU', and 'DA-GRU'.
    optimized_model : custom neural network class
        The trained model from the best-performing optuna trial in the study
    test_results : dict
        Dictionary containing the results for the neural network testing
        protocol. Additionally, the target and model prediction time series are
        stored within this dictionary. 

    Returns
    -------
    None.

    """
    
    # Define results file directory
    project_directory = os.path.dirname(os.getcwd())
    results_directory = os.path.join(project_directory, 'results')

    # Save neural network
    save_neural_network(model_type, optimized_model, results_directory)
    
    # Save test results
    test_results_filename = os.path.join(results_directory, f'{model_type}_test_results.pickle')
    with open(test_results_filename, 'wb') as fout:
        pickle.dump(test_results, fout)  
        
    return