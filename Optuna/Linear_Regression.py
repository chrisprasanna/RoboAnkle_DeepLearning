"""
Created on Fri Oct 22 12:59:38 2021

@author: Chris Prasanna

Tools for simple polynomial fit (linear regression)
"""
#%% Imports

import numpy as np
from sklearn.model_selection import KFold
from scipy import optimize
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as tick

#%% Nominal Values
h = 0.134
d = 0.06
eta = 0.6
b = np.sqrt(d**2 + h**2) 
theta0 = np.deg2rad(-4.5)
z1 = b**2 + d**2
z2 = -2*b*d
R1 = 19
R2 = 2*np.pi/0.004
k_tau = 0.00775
J_m = (4.09/1000)/(100^2)
J_g = (0.4/1000)/(100^2)
J_f = J_m / 10000
R_g = 3249/169
B = 0.01
mass = 85

#%% Functions
def f(x, a, b, c, d):
    # third-degree nonlinear (polynomial) model 
    return a*x**3 + b*x**2 + c*x + d

def residual(p, x, y):
    return y - f(x, *p)

def fbeta(theta):
    beta = np.arccos(d/b) + theta - theta0
    return beta

def fa(theta):
    a = np.sqrt(b**2 + d**2 - 2*b*d*np.cos(fbeta(theta) + theta - theta0));
    return a

def falpha(theta):
    alpha = np.arcsin(np.divide(d*np.sin(fbeta(theta)),fa(theta)));
    return alpha

def fphi(theta):
    phi = np.pi/2 - fbeta(theta) - falpha(theta)
    return phi

def transmission(theta):
    R = d*np.cos(fphi(theta))*R1*R2*eta
    return R

def drivetrain_model(theta, current):
    tau_a = (k_tau/mass)*np.multiply(transmission(theta),current)
    return tau_a

def plot_full(x, im, y, constants, best_model):
    
    PATH = constants['results path']
    num_trials = constants['number of walking trials']
    fs = constants['sub-sample freq']
    
    num_timestepsPerTrial = constants['number of timesteps per trial']
    timesteps = num_timestepsPerTrial - 1
    tr_percent = 0.70
    val_percent = 0.15
    test_percent = 0.15
    train_length = int(np.round(timesteps*tr_percent))
    test_length = int(np.floor(timesteps*test_percent))
    val_length = timesteps - train_length - test_length
    
    x_train = x[:,:train_length]
    x_val = x[:,train_length:train_length+val_length]
    x_test = x[:,train_length+val_length:]
    im_train = im[:,:train_length]
    im_val = im[:,train_length:train_length+val_length]
    im_test = im[:,train_length+val_length:]
    y_train = y[:,:train_length]
    y_val = y[:,train_length:train_length+val_length]
    y_test = y[:,train_length+val_length:]
    
    for i in range(0,num_trials):
        # True Vals
        train_true = y_train[i,:]
        val_true = y_val[i,:]
        test_true = y_test[i,:]
        true_vals = np.concatenate((train_true, val_true, test_true), axis=0)
        
        # Model outputs
        train_out = f(x_train[i,:], *best_model) + drivetrain_model(x_train[i,:], im_train[i,:])
        val_out = f(x_val[i,:], *best_model) + drivetrain_model(x_val[i,:], im_val[i,:])
        test_out = f(x_test[i,:], *best_model) + drivetrain_model(x_test[i,:], im_test[i,:])
        
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
        plt.title(f'Polynomial Model: Full Trial {i+1}',fontsize=30)
        plt.xlim(0,30) 
        plt.grid(True)
        plt.legend(prop={"size":20}, loc='upper left')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        # Save
        filename = f'Polynomial Model -- Full Trial{i+1}.png'
        savepath = os.path.join(PATH, 'Polynomial' , filename) 
        fig.savefig(savepath, bbox_inches='tight')
    
    return

def train_poly(constants):
    
    PATH = constants['results path']
    num_trials = constants['number of walking trials']
    
    data = constants['dataset']
    features = data['Features']
    response_ = data['Responses']
    
    # Dataset splitting params   
    num_timestepsPerTrial = constants['number of timesteps per trial']
    timesteps = num_timestepsPerTrial - 1
    tr_percent = 0.70
    val_percent = 0.15
    test_percent = 0.15
    train_length = int(np.round(timesteps*tr_percent))
    test_length = int(np.floor(timesteps*test_percent))
    val_length = timesteps - train_length - test_length
    
    # Get key features
    ankle_angles_ = features[:,:,3] # CHECK INDEX TO MAKE SURE THIS IS CORRECT
    current_ = features[:,:,0]
    
    # Split dataset
    ankle_angles = ankle_angles_[:,:train_length+val_length]
    current = current_[:,:train_length+val_length]
    response = response_[:,:train_length+val_length]
    
    # Flatten
    ankle_angles = ankle_angles.flatten()
    current = current.flatten()
    response = response.flatten()
    
    print('Nonlinear Least-Squares')
    
    # k-fold cross val params
    k = 5
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    
    # pre-allocate    
    mse = np.zeros(k)
    p = np.zeros((k,4))
    
    # loop
    ii = 0
    for train_index, test_index in kf.split(response):
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        # Train
        p0 = [1., 1., 1., 1.]
        x = ankle_angles[train_index]
        y = response[train_index] - drivetrain_model(x, current[train_index])
        popt, pcov = optimize.leastsq(residual, p0, args=(x, y))
        
        # Val
        xn = ankle_angles[test_index]
        y_true = response[test_index]
        y_pred = f(xn, *popt) + drivetrain_model(ankle_angles[test_index], current[test_index])
        mse[ii] = mean_squared_error(y_true, y_pred, squared=True)       
        p[ii,:] = popt
        ii+=1 
        
    best_model = p[np.argmin(mse),:]
    
    # Test
    ankle_angles_test = ankle_angles_[:,train_length+val_length:]
    current_test = current_[:,train_length+val_length:]
    response_test = response_[:,train_length+val_length:]
    
    ankle_angles = ankle_angles_test.flatten()
    current = current_test.flatten()
    response = response_test.flatten()
    
    pred = f(ankle_angles, *best_model) + drivetrain_model(ankle_angles, current)
    test_rmse = mean_squared_error(response, pred, squared=False)  
    test_pcc , _ = pearsonr(response, pred)
    
    # save results
    filename = os.path.join(PATH, 'Polynomial', 'Optimization_Results.txt')
    print('tau = (p1*x^3 + p2*x^2 + p3^x + p4) + (k/m)*R*i \n\n', file=open(filename, "a"))
    print('Coefficients: ', file=open(filename, "a"))
    print('\t\n'.join(map(str, best_model)), file=open(filename, "a"))
    print('\nRMSE: {:.6f} Nm/kg\n'.format(test_rmse), file=open(filename, "a"))
    print('PCC: {:.6f}\n'.format(test_pcc), file=open(filename, "a"))
    
    # Visualize - Time Series
    for i in range(0,num_trials):
        y = response_test[i,:]
        x = ankle_angles_test[i,:]
        im = current_test[i,:]
        t = np.linspace(start=0,stop=len(y)/30,num=len(y))
        
        yp = f(x, *best_model) + drivetrain_model(x, im)
        
        fig = plt.figure(figsize=(20,10))
        plt.plot(t , y, label='True', color='black',linewidth=4)
        plt.plot(t , yp, label='Predicted', color='red',linewidth=4)
        plt.xlabel('Time [s]',fontsize=20)
        plt.ylabel('Moment [Nm/kg]',fontsize=20)
        plt.xlim(0,t[-1]) # consistent scale
        plt.grid(True)
        plt.legend(prop={"size":20}, loc='upper left')
        plt.title(f'Polynomial Model: Test Trial {i+1}',fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        # Save
        filename = f'Polynomial -- Test{i+1}.png'
        savepath = os.path.join(PATH, 'Polynomial' , filename) 
        fig.savefig(savepath, bbox_inches='tight')
        
    # Visualize Histogram
    err = response - pred
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
    plt.title(f'Polynomial Testing Errors: ' + str(meanErr) + ' +/- ' + str(stdErr) + ' Nm/kg',fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    # Save
    filename = f'Polynomial -- Error Histogram.png'
    savepath = os.path.join(PATH, 'Polynomial' , filename) 
    fig.savefig(savepath, bbox_inches='tight')
    
    # Visualize Full Trials
    plot_full(ankle_angles_, current_, response_, constants, best_model)

    return best_model, test_rmse, response, pred