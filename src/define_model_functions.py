"""

This file contains functions related to initializing and defining model 
architectures to be trained and evaluated. 
The primary purpose of all of these functions is to return a class object 
defining the model to our primary training and analysis scripts.

"""

from src import neural_network_classes


def get_neural_network(model_type, hyperparameters, constants):
    """
    This function defines a neural network model using one of three custom 
    class objects (found in neural_network_classes.py), hyperparameter values, 
    and constant information. This function returns the untrained neural 
    network object which is then passed on to training and evaluation scripts. 

    Parameters
    ----------
    model_type : string
        The type of neural network you wish to define. The three options are 
        'FFN', 'GRU', and 'DA-GRU'.
    hyperparameters : dict
        This dictionary has the name of each hyperparameter as keys and their 
        value. Hyperparameters are chosen by the user or optimized via a framework 
        such as Optuna in order to improve neural network prediction 
        performance. Note that some hyperparameters are only used for specific 
        DNN types.
    constants : dict
        This dictionary has the name of each constant as keys and their values. 
        This dict contains constant information to be passed across 
        various DNN training and evaluation functions. The data within this 
        dict is not meant to be changed after initialization (i.e., these are 
        values that Optuna does not optimize) and are specific to the modeling
        task (e.g., input size, output size, computing device, etc.).

    Returns
    -------
    nn : torch.nn.Module
        Custom neural network torch class containting architecture information 
        and forward pass functionality 

    """

    if (model_type == 'FFN'):
        nn = neural_network_classes.FFN(hyperparameters, constants)
    elif(model_type == 'GRU'):
        nn = neural_network_classes.GRU(hyperparameters, constants)
    elif(model_type == 'DA-GRU'):
        nn = neural_network_classes.DAGRU(hyperparameters, constants)
    else:
        print("arguemnt 'model_type' should be one of the following: FFN, GRU, DA-GRU")
    return nn
