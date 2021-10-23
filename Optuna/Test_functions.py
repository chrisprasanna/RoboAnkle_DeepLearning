# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:58:18 2021

Functions to Test each NN

@author: cpras
"""

#%% Imports

import torch

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import scipy as scipy
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as tick

#%% Evaluate the Networks on the Test Dataset

def test_network(test_loader, model, model_name, criterion, PATH, device):
    
    test_length = test_loader.dataset.data.shape[1]
    gru_flag = False
    
    model.eval()

    test_loss = 0.0
    RMSE = []
    PCC = []
    
    target = []
    pred = []
    
    # ***************
    # Note: include code that turns gradients off to save memory
    # >> is_train = False
    # >> with torch.set_grad_enabled(is_train):
    # OR if that doesn't work
    # >> torch.set_grad_enabled(False)
    # >> with torch.no_grad():
    #     
    # Be sure to also remove the device dependencies as well
    # ***************
    with torch.no_grad():
    
        for (idx, batch) in enumerate(test_loader):
            
            batch_x = batch[0].squeeze().to(device)
            batch_y = batch[1].squeeze().to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            if model_name == 'FFN':
                output = model(batch_x.float())
                
            elif model_name == 'GRU':
                if gru_flag == False:
                    h = model.init_hidden(batch_size = test_length)
                    gru_flag = True
                    
                output,h = model(batch_x.float(),h)
                output = output[:,-1,0]
                
            elif model_name == 'Transformer':
                output = model(batch_x.float())
                output = output[:,-1,0]
                
            elif model_name == 'DA-RNN':
                output = model(batch_x.float())
                
            else:
                print('Model Name is not one of the four options!!')
                print('Must be FFN, GRU, Transformer, or DA-RNN')
                model = []
            
            # calculate the loss
            yPred = output.squeeze().cpu()
            yTrue = batch_y.float().cpu()   
            loss = criterion(yPred, yTrue)
            # update test loss 
            test_loss += loss.item()*yPred.size(0)
            # compare predictions to true label
            rms = mean_squared_error(yTrue.detach().numpy(), yPred.detach().numpy(), squared=False)
            RMSE.append(rms)
            pcc , _ = pearsonr(yTrue.detach().numpy(), yPred.detach().numpy())
            PCC.append(pcc)
            # for plotting
            target.append(yTrue.detach().numpy())
            pred.append(yPred.detach().numpy())
            
            # delete intermediate values
            del batch_x, batch_y, output, yPred, yTrue, loss
        
    # calculate and print avg test loss
    filename = os.path.join(PATH, model_name, 'Optimization_Results.txt')
    test_loss = test_loss/test_length
    print('\nTest Loss: {:.6f}\n'.format(test_loss), file=open(filename, "a"))
    print('RMSE: {:.6f} Nm/kg\n'.format(np.average(RMSE)), file=open(filename, "a"))
    print('PCC: {:.6f}\n'.format(np.average(PCC)), file=open(filename, "a"))
    
    return target, pred, np.average(RMSE), test_loss, np.average(PCC)

#%% Visualize the Test Results

def visualize_results(model, model_name, train_loader, val_loader, test_loader, 
                      num_trials, target, pred, PATH, fs, device):
    
    train_length = train_loader.dataset.data.shape[1]
    val_length = val_loader.dataset.data.shape[1]
    test_length = test_loader.dataset.data.shape[1]
    
    Targets = np.transpose(np.array(target))
    Predictions = np.transpose(np.array(pred))
    t = np.linspace(start=0,stop=len(Targets)/fs,num=len(Targets))
    
    ######################
    # Test Data Plots
    ######################
    for i in range(0,num_trials):
        fig = plt.figure(figsize=(20,10))
        plt.plot(t , Targets[:,i], label='True', color='black',linewidth=4)
        plt.plot(t , Predictions[:,i], label='Predicted', color='red',linewidth=4)
        plt.xlabel('Time [s]',fontsize=20)
        plt.ylabel('Moment [Nm/kg]',fontsize=20)
        plt.xlim(0,t[-1]) # consistent scale
        plt.grid(True)
        plt.legend(prop={"size":20}, loc='upper left')
        plt.title(f'{model_name}: Test Trial {i+1}',fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        # Save
        filename = f'{model_name}-- Test{i+1}.png'
        savepath = os.path.join(PATH, model_name , filename) 
        fig.savefig(savepath, bbox_inches='tight')
    
    ######################
    # Error Histogram
    ######################
    T = np.ndarray.flatten(Targets)
    P = np.ndarray.flatten(Predictions)
    err = T-P
    meanErr = round(np.mean(err),4)
    stdErr = round(np.std(err),4)
    bins = 30
     
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    N, bins, patches = ax.hist(err, bins=bins, density=True, histtype='bar')
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    # Now we format the y-axis to display percentage
    ax.yaxis.set_major_formatter(tick.PercentFormatter())
    
    plt.xlabel('Error [Nm/kg]',fontsize=20)
    plt.title(f'{model_name} Testing Errors: ' + str(meanErr) + ' +/- ' + str(stdErr) + ' Nm/kg',fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    # Save
    filename = f'{model_name}-- Error Histogram.png'
    savepath = os.path.join(PATH, model_name , filename) 
    fig.savefig(savepath, bbox_inches='tight')
    
    # Percent Error and Fit
    percentError = mean_absolute_error(T, P)*100
    percentFit = 100 - percentError
    
    ######################
    # Full Trials
    ######################
    
    model.eval()
    
    # ***************
    # Note: include code that turns gradients off to save memory
    # >> is_train = False
    # >> with torch.set_grad_enabled(is_train):
    # OR if that doesn't work
    # >> torch.set_grad_enabled(False)
    # >> with torch.no_grad():
    #     
    # Be sure to also remove the device dependencies as well
    # ***************
    
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    test_iterator = iter(test_loader)
    for i in range(0,num_trials):
        # print(f'--- Full Test Trial {i+1}')
        try:
            train_x, train_y = next(train_iterator)
            val_x, val_y = next(val_iterator)
            test_x, test_y = next(test_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            val_iterator = iter(val_loader)
            test_iterator = iter(test_loader)
            
            train_x, train_y = next(train_iterator)
            val_x, val_y = next(val_iterator)
            test_x, test_y = next(test_iterator)
        
        # Model Predictions
        if model_name == 'FFN':
            train_out = model(torch.flatten(train_x,start_dim=2).squeeze().to(device).float())
            val_out = model(torch.flatten(val_x,start_dim=2).squeeze().to(device).float())
            test_out = model(torch.flatten(test_x,start_dim=2).squeeze().to(device).float())
            train_out = train_out.cpu().detach().numpy()
            val_out = val_out.cpu().detach().numpy()
            test_out = test_out.cpu().detach().numpy()
            
        elif model_name == 'GRU':
            h_tr = model.init_hidden(batch_size = train_length).to(device)
            h_val = model.init_hidden(batch_size = val_length).to(device)
            h_test = model.init_hidden(batch_size = test_length).to(device)
            
            train_out, _ = model(train_x.squeeze().to(device).float(),h_tr)
            val_out, _ = model(val_x.squeeze().to(device).float(),h_val)
            test_out, _ = model(test_x.squeeze().to(device).float(),h_test)
            train_out = train_out[:,-1,0].cpu().detach().numpy()
            val_out = val_out[:,-1,0].cpu().detach().numpy()
            test_out = test_out[:,-1,0].cpu().detach().numpy()
            
        elif model_name == 'Transformer':
            train_out = model(train_x.squeeze().to(device).float())
            val_out = model(val_x.squeeze().to(device).float())
            test_out = model(test_x.squeeze().to(device).float())
            train_out = train_out[:,-1,0].cpu().detach().numpy()
            val_out = val_out[:,-1,0].cpu().detach().numpy()
            test_out = test_out[:,-1,0].cpu().detach().numpy()
            
        elif model_name == 'DA-RNN':
            train_out = model(train_x.squeeze().to(device).float()).cpu().detach().numpy()
            val_out = model(val_x.squeeze().to(device).float()).cpu().detach().numpy()
            test_out = model(test_x.squeeze().to(device).float()).cpu().detach().numpy()
            
        else:
            print('Model Name is not one of the four options!!')
            print('Must be FFN, GRU, Transformer, or DA-RNN')
            model = []
        
        # True Vals
        train_true = train_y.squeeze().float().detach().numpy()
        val_true = val_y.squeeze().float().detach().numpy()
        test_true = test_y.squeeze().float().detach().numpy()
        true_vals = np.concatenate((train_true, val_true, test_true), axis=0)
        
        # Time vectors
        time = np.linspace(start=30-len(true_vals)/fs,stop=30,num=len(true_vals))
        time_tr = time[:train_length]
        time_val = time[train_length:train_length+val_length]
        time_test = time[train_length+val_length:]
        
        # Plot
        fig = plt.figure(figsize=(20,10))
        plt.plot(time , true_vals, color='black', label='True',linewidth=4)
        plt.plot(time_tr , train_out, color='blue', label='Train',linewidth=4)
        plt.plot(time_val , val_out, color='green', label='Validation',linewidth=4)
        plt.plot(time_test , test_out, color='red', label='Test',linewidth=4)
        plt.xlabel('Time [s]',fontsize=20)
        plt.ylabel('Moment [Nm/kg]',fontsize=20)
        plt.title(f'{model_name}: Full Trial {i+1}',fontsize=30)
        plt.xlim(0,30) 
        plt.grid(True)
        plt.legend(prop={"size":20}, loc='upper left')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        # Save
        filename = f'{model_name}-- Full Trial{i+1}.png'
        savepath = os.path.join(PATH, model_name , filename) 
        fig.savefig(savepath, bbox_inches='tight')
    
    
    return percentFit, percentError

#%% Fitted Histograms

def fitted_histogram(target1, pred1, target2, pred2, target3, pred3, target4, pred4, PATH):
    
    # with polynomial     
    fig = plt.figure(figsize=(20,10))
    bins = 30
    
    best_fit_line1, bins1 =  fit_line_calc(target1, pred1, bins, 'blue')
    best_fit_line2, bins2 =  fit_line_calc(target2, pred2, bins, 'green')
    best_fit_line3, bins3 =  fit_line_calc(target3, pred3, bins, 'orange')
    best_fit_line4, bins4 =  fit_line_calc(target4, pred4, bins, 'red')
    
    plt.plot(bins4, best_fit_line4, label='Polynomial',color='red', linewidth=4)
    plt.plot(bins1, best_fit_line1, label='FFN',color='blue', linewidth=4)
    plt.plot(bins2, best_fit_line2, label='GRU',color='green', linewidth=4)
    plt.plot(bins3, best_fit_line3, label='DA-RNN',color='orange', linewidth=4)
    plt.legend(prop={"size":20}, loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Error [Nm/kg]',fontsize=20)
    plt.ylabel('Probability',fontsize=20)
    plt.title('Error Distribution for all Models',fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    # Save
    filename = 'Error Histogram - All Models.png'
    savepath = os.path.join(PATH, filename) 
    fig.savefig(savepath, bbox_inches='tight')
    
    # just NNs
    fig = plt.figure(figsize=(20,10))
    bins = 30
    
    best_fit_line1, bins1 =  fit_line_calc(target1, pred1, bins, 'blue')
    best_fit_line2, bins2 =  fit_line_calc(target2, pred2, bins, 'green')
    best_fit_line3, bins3 =  fit_line_calc(target3, pred3, bins, 'orange')
    
    plt.plot(bins1, best_fit_line1, label='FFN',color='blue', linewidth=4)
    plt.plot(bins2, best_fit_line2, label='GRU',color='green', linewidth=4)
    plt.plot(bins3, best_fit_line3, label='DA-RNN',color='orange', linewidth=4)
    plt.legend(prop={"size":20}, loc='upper left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Error [Nm/kg]',fontsize=20)
    plt.ylabel('Probability',fontsize=20)
    plt.title('Error Distribution for all Models',fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    # Save
    filename = 'Error Histogram - All NNs.png'
    savepath = os.path.join(PATH, filename) 
    fig.savefig(savepath, bbox_inches='tight')
    
    return

def fit_line_calc(target, pred, bins, c):
        
    Targets = np.transpose(np.array(target))
    Predictions = np.transpose(np.array(pred))
    T = np.ndarray.flatten(Targets)
    P = np.ndarray.flatten(Predictions)
    err = T-P
    
    _, bins, _ = plt.hist(err, 20, density=1, alpha=0.5, color=c)
    
    mu, sigma = scipy.stats.norm.fit(err)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    
    return best_fit_line, bins