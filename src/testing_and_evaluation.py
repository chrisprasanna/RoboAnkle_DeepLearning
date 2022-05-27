"""
Created on Thu Apr 28 11:24:30 2022

@author: Chris Prasanna
"""

from src import data_processing

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# %% 

def preprocess_data_for_testing(data_set, sequence_length, constants):
    """
    This function defines the pipeline that processes the data samples and 
    prepares them for testing. Sub-functions represent different
    data processing procedures and are defined in data_processing.py.

    Parameters
    ----------
    data_set : pytorch dataloader object
        Contains all files used for the testing dataset, the sample of data 
        used to evalaute the model. File names map to pandas dataframes.
    sequence_length : int
        Sequence Length is the length of the historical sequence of input data. 
        If the sequence length is N, then the window range of the input data 
        will be from samples [0:-1,-2,...,-N].
    constants : dict
        This dictionary has the name of each constant as keys and their values.

    Returns
    -------
    processed_data_set : torch dataloader class
        Processed dataset converted into a PyTorch DataLoader that can iterate 
        through the data during training and evaluation as needed

    """  
    # Resample data
    # NOTE: this could be used outside Optuna so it's not called repeatedly
    resampled_data = data_processing.resample_data(data_set, constants)
    
    # Build lookback window
    features, targets = data_processing.get_lookback_windows(resampled_data, sequence_length, constants)
    
    # Data normalization
    normalized_features, normalized_targets = data_processing.normalize_data(features, targets, constants, is_training_set=False)
    
    # Convert data to tensors
    tensor_features, tensor_targets = data_processing.convert_to_tensors(normalized_features, normalized_targets)
    
    # Create datasets
    torch_data_set = data_processing.create_torch_datasets(tensor_features, tensor_targets, train_bool=False)
    
    # Create Dataloaders
    processed_data_set = torch.utils.data.DataLoader(torch_data_set,batch_size=1,
                                               shuffle=False, drop_last=True)
    
    return processed_data_set

#%% 

def compute_test_metrics(test_results, targets, model_prediction, trial, target_signal_names):
    """
    This function updates the test_results dictionary to include additional
    test metrics computed from the target and model prediction output signals.
    First, the two types of signals are stored within the dictionary. Next, 
    RMSE and PCC are computed for each walking trial within the test dataset. 

    Parameters
    ----------
    test_results : dict
        Dictionary containing the results for the neural network testing
        protocol. Additionally, the target and model prediction time series are
        stored within this dictionary.
    targets : torch tensor
        Target series from the tests dataset to be compared to the model 
        prediction outputs
    model_prediction : torch tensor
        Model prediction outputs from the test dataset
    trial : string
        Walking trial within the test dataset that will be assessed
    target_signal_names : list
        List where each element contains a string representing each model
        output / target signal.

    Returns
    -------
    None.

    """
    # Convert targets and predictions to numpy and record
    Targets = targets.detach().numpy()
    Predictions = model_prediction.detach().numpy()
    test_results[trial]['Targets'] = pd.DataFrame(Targets, columns=target_signal_names)
    test_results[trial]['Predictions'] = pd.DataFrame(Predictions, columns=target_signal_names)
    
    # Compute RMSE
    rmse = mean_squared_error(Targets, Predictions, squared=False, multioutput='raw_values')
    test_results[trial]['RMSE'] = pd.DataFrame([rmse], columns=target_signal_names)
    
    # Compute PCC
    pcc = np.zeros(len(target_signal_names))
    for idx in range(len(target_signal_names)):
        output_name = target_signal_names[idx] 
        output_target = test_results[trial]['Targets'][output_name]
        output_prediction = test_results[trial]['Predictions'][output_name]
        pcc[idx] , _ = pearsonr(output_target, output_prediction)        
    test_results[trial]['PCC'] = pd.DataFrame([pcc], columns=target_signal_names)
    
    # Delete intermediate values
    del pcc, rmse, Targets, Predictions
    
    return

#%% 

def test_neural_network(model, model_type, test_set, constants, criterion, test_walking_trials, target_signal_names):
    """
    This function defines the neural network testing loop. This function also
    de-normalizes the output and target signals. A function that computes 
    additional test metrics is also included in this function definition. 

    Parameters
    ----------
    model : custom neural network class
        The trained model from the best-performing optuna trial in the study
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    test_set : pytorch dataloader object
        Contains all files used for the testing dataset, the sample of data 
        used to evalaute the model. File names map to pandas dataframes.
    constants : dict
        This dictionary has the name of each constant as keys and their values.
    criterion : torch.nn Loss Function
        A serializable function where, given an input and a target, computes a 
        gradient according to a given loss function. 
    test_walking_trials : list
        List where each element contains a string representing each walking 
        trial contained within the test dataset. 
    target_signal_names : list
        List where each element contains a string representing each model
        output / target signal. 

    Returns
    -------
    test_results : dict
        Dictionary containing the results for the neural network testing
        protocol. Additionally, the target and model prediction time series are
        stored within this dictionary.

    """   
    # Unpack constants
    device = constants['device']
    
    # pre-allocate
    test_results = {}
    
    # Retrieve maximum and minimum target values for denormalization
    targets_max = torch.Tensor(constants['Maximum Targets'])
    targets_min = torch.Tensor(constants['Minimum Targets'])
    
    # prep model for evaluation
    model.eval()
    with torch.no_grad():
        # Loop through test batches
        for (idx, batch) in enumerate(test_set):           
            # Define current trial and preallocate dict
            trial = test_walking_trials[idx]
            test_results[trial] = {}
            
            # If GRU, initialize hidden state
            if model_type== 'GRU':
                h_test = model.init_hidden(batch_size = batch[0].shape[1])
            
            # Extract features and targets
            features = batch[0].squeeze().to(device)
            targets = batch[1].squeeze().to(device)
            
            # Forward pass
            if model_type == 'GRU':
                model_prediction,h_test = model(features.float(),h_test)
                model_prediction = model_prediction[:,-1,:]
                h_test = h_test.detach() # detach hidden in between batches
            else:
                model_prediction = model(features.float())
        
            # Compute loss and record
            loss_object = criterion() # instantiate loss class
            loss = loss_object(model_prediction, targets.float()) # use object to compute loss            
            test_results[trial]['loss'] = loss.item()
            
            # De-normalize targets & model predictions
            targets = (targets.float()*(targets_max - targets_min)) + targets_min
            model_prediction = (model_prediction*(targets_max - targets_min)) + targets_min
            
            # Record test signals and metrics
            compute_test_metrics(test_results, targets, model_prediction, trial, target_signal_names)
            
            # Delete intermediate variables and empty cache
            del features, targets, model_prediction, loss
            torch.cuda.empty_cache() 
    
    return test_results

# %% 

def generate_figures_in_multipage_pdf(filename, figs):
    """
    This function generates a multi-page pdf containing each of the test result
    figures. 

    Parameters
    ----------
    filename : string
        Filename and path to save the pdf containing time series prediction 
        figures
    figs : array of matplotlib.pyplot.figure objects, optional
        Array where each element contains a figure identifier

    Returns
    -------
    None.

    """
    pp = PdfPages(filename)
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    return

# %%

def visualize_COBRA_test_results(model_type, test_results, test_walking_trials, target_signal_names, sequence_length):
    """
    This function generates time series plots of the target signals and model
    prediction output signals. Each figure contains a single output variable. 
    Each figure holds subplots, each representing a different walking trial
    within the test dataset. Figures are then saved locally. 

    Parameters
    ----------
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    test_results : dict
        Dictionary containing the results for the neural network testing
        protocol. Additionally, the target and model prediction time series are
        stored within this dictionary.
    test_walking_trials : list
        List where each element contains a string representing each walking 
        trial contained within the test dataset. 
    target_signal_names : list
        List where each element contains a string representing each model
        output / target signal. 
    sequence_length : int
        Sequence Length is the length of the historical sequence of input data. 
        If the sequence length is N, then the window range of the input data 
        will be from samples [0:-1,-2,...,-N].

    Returns
    -------
    None.

    """
    # Define order to loop through subplots
    subplot_indices = [[0,0], [0,1], [1,0], [1,1]]
    
    # Pre-allocate
    figs = []
    
    # Create subplots of time series predictions
    for output_idx in range(len(target_signal_names)):
        
        # Get output signal
        output = target_signal_names[output_idx]
        
        # Set up figure
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(20,12))
        
        # Loop through test trials
        for trial_idx in range(len(test_walking_trials)):
            
            # Define trial
            trial = test_walking_trials[trial_idx]
            
            # Retrieve targets and predictions
            targets = test_results[trial]['Targets']
            predictions = test_results[trial]['Predictions']
            
            # Define subplot
            subplot_idx = subplot_indices[trial_idx]
            
            # Define time vector
            total_timesteps = len(predictions[output]) + sequence_length + 1
            delta_time = 30 / total_timesteps
            time = np.linspace(start=delta_time*sequence_length, stop=30, num=len(predictions[output]))
            
            # Plot
            ax = axs[subplot_idx[0], subplot_idx[1]]
            ax.plot(time, targets[output], color='black', label='Target', linewidth=2)
            ax.plot(time, predictions[output], color='blue', label='Prediction', linewidth=2)
            ax.set_title(f'{trial}',fontsize=25)
            ax.grid(True)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
        
        # Common titles
        fig.suptitle(f'{model_type} {output} Predictions on Test Dataset',fontsize=30)
        fig.supxlabel('Time (s)', fontsize=25)
        fig.supylabel(f'{output}', fontsize=25)
        # Legend
        lines, labels = fig.axes[-1].get_legend_handles_labels()    
        fig.legend(lines, labels, loc='lower right', fontsize=20, ncol=2)
        # Format
        fig.tight_layout(rect=[0.02, 0.03, 0.95, 0.97])
        plt.show()
        # Collect figure
        figs.append(fig)
    
    # Define results file directory
    project_directory = os.path.dirname(os.getcwd())
    results_directory = os.path.join(project_directory, 'results')
    
    # Save figures
    figure_filename = os.path.join(results_directory, f'{model_type}_test_figures.pdf')
    generate_figures_in_multipage_pdf(figure_filename, figs)
    
    return

# %% 

def main_test(model, model_type, data_set, constants, sequence_length):
    """
    This function defines the main neural network testing protocol. First, the 
    walking trials and output variables are defined. Next, the data is processed 
    and prepared for testing. Next, the models are tested. Finally, the test
    results are visualized. 

    Parameters
    ----------
    model : custom neural network class
        The trained model from the best-performing optuna trial in the study
    model_type : string
        The type of model that is being trained for this optuna study. The three 
        options are 'FFN', 'GRU', and 'DA-GRU'.
    data_set : pytorch dataloader object
        Contains all files used for the testing dataset, the sample of data 
        used to evalaute the model. File names map to pandas dataframes. 
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
    sequence_length : int
        Sequence Length is the length of the historical sequence of input data. 
        If the sequence length is N, then the window range of the input data 
        will be from samples [0:-1,-2,...,-N].

    Returns
    -------
    None.

    """  
    # Get walking trial names
    test_walking_trials = list(data_set.keys())
    
    # Output target column names
    target_signal_names = ['PAFP Torque (Nm/kg)']
    
    # Preprocess test dataset
    processed_test_set = preprocess_data_for_testing(data_set, sequence_length, constants)
    
    # Define criterion / cost function
    criterion = torch.nn.MSELoss
    
    # Compute test results
    test_results = test_neural_network(model, model_type, processed_test_set, constants, criterion, test_walking_trials, target_signal_names)
    
    # Visualize Results
    visualize_COBRA_test_results(model_type, test_results, test_walking_trials, target_signal_names, sequence_length)
    
    return test_results