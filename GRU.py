"""
Created on Tue Jan 26 17:04:25 2021
GRU Prosthetic Ankle Torque Predictor 

@author: Chris Prasanna
"""
#%% Imports

import torch
import torch.nn as nn
import torch.optim as optim  # ADAM, SGD, etc
import torch.nn.functional as F  # relu, tanh, etc.
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import time
import scipy.io as spio
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as tick

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import torch.onnx
import onnx
from onnx_tf.backend import prepare

#%% Classes
    
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
    
# Create a GRU Network
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_length,dropout):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, 
                          num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
     
    def forward(self, x, h0):
        out, h0 = self.gru(x, h0)
        out = self.fc(self.relu(out))
        return out, h0
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#%% Functions
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict  

#Defining MAPE function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

#%% Load and Organize Data

# Load Data from .mat file
print('Loading Data...')
matfile = r'JR_data_ankleTorque.mat'
matdata = loadmat(matfile)
Data = matdata['Data']
print('Loading Complete')

# Size of Data
dims = Data['Features'].shape
num_trials = dims[0]
num_timestepsPerTrial = dims[1]
num_features = dims[2]

active_features = Data['activeFeatures']
passive_features = Data['passiveFeatures']
num_activeTrials = active_features.shape[0]
num_passiveTrials = passive_features.shape[0]

# Inputs and Outputs
features = Data['Features']
responses = Data['Responses']

#%% Hyperparameters

# Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DNN Hyperparameters
input_size = num_features
output_size = 1
num_layers = 2
hidden_size = 256       # [4,8,16,32,64,128,256,512]
sequence_length = 11

step_size = 1
batch_factor = 10 # How to divide time dimension into batch (trial) dimension 
# seqMode = torch.Tensor(np.asarray([0]))    # 1 = seq2seq, 0 = seq2one

# Optimizer Hyperparameters
learning_rate = 0.001   # inital LR
lr_step_size = 3           # number of epochs before reducing learning rate
gamma = 0.5             # learning rate decrease factor
weight_decay = 0.05     # L2Regularization (default = 0); tested: 1e-5, 0.05
amsgrad = False

# Training Hyperparameters
num_epochs = 1000
batch_size = num_trials
noiseSTD = 0.2
dropout = 0.2

#%% Temporal Processing

print('Processing Time Series Data...')

X = np.zeros((num_trials, num_timestepsPerTrial, sequence_length, input_size))
y = np.zeros((num_trials, num_timestepsPerTrial, sequence_length, output_size))
target = np.zeros((num_trials, num_timestepsPerTrial, output_size))

for n in range(num_trials):

    data_x = pd.DataFrame(features[n,:,:],
                       columns=['im', 'wm','dTh_h','Th_h','dTh_a','Th_a','Fh_l','Fh_r','Fm_l','Fm_r','Ft_l','Ft_r'])
    data_y = pd.DataFrame(responses[n,:], columns=['tau'])
    
    for i, name in enumerate(list(data_x.columns)):
        for j in range(sequence_length):
            X[n, :, j, i] = data_x[name].shift(sequence_length - j - 1).fillna(method="bfill")
            
    for j in range(sequence_length):
        y[n,:,j,0] = data_y["tau"].shift(sequence_length - j - 1).fillna(method="bfill")
            
    prediction_horizon = 1
    target[n,:,0] = data_y["tau"].shift(-prediction_horizon).fillna(method="ffill").values

    
X = X[:,sequence_length:-1]
y = y[:,sequence_length:-1]
target = target[:,sequence_length:-1]

#%% Split Dataset - Train, Val, Test

print('Splitting Data into Training, Validation, and Test Sets...')

timesteps = num_timestepsPerTrial - sequence_length - 1

tr_percent = 0.70
val_percent = 0.15
test_percent = 0.15

train_length = int(np.round(timesteps*tr_percent))
test_length = int(np.floor(timesteps*test_percent))
val_length = timesteps - train_length - test_length

X_train = X[:,:train_length]
X_val = X[:,train_length:train_length+val_length]
X_test = X[:,train_length+val_length:] # X[:,-val_length:]
target_train = target[:,:train_length]
target_val = target[:,train_length:train_length+val_length]
target_test = target[:,train_length+val_length:] # target[:,-val_length:]

#%% Data Scaling

print('Scaling Data from [0,1]...')

X_train_max = X_train.max(axis=(0,1,2))
X_train_min = X_train.min(axis=(0,1,2))
target_train_max = target_train.max(axis=(0,1))
target_train_min = target_train.min(axis=(0,1))

X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

# target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
# target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
# target_test = (target_test - target_train_min) / (target_train_max - target_train_min)

#%% Create Datasets and Loaders

print('Create Pytorch Data Loaders...')

X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
target_train_t = torch.Tensor(target_train)
target_val_t = torch.Tensor(target_val)
target_test_t = torch.Tensor(target_test)

# Add Guassian Noise to Training and Validation Inputs
noiseBoolean = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]) # last 4 inputs already are noisy signals
Xtrain_noisy = X_train_t  + noiseSTD*torch.randn(X_train_t.size())*noiseBoolean
Xval_noisy = X_val_t + noiseSTD*torch.randn(X_val_t.size())*noiseBoolean

train_dataset = CustomTensorDataset([Xtrain_noisy, target_train_t])
val_dataset = CustomTensorDataset([X_val_t, target_val_t])
test_dataset = CustomTensorDataset([X_test_t, target_test_t])                              

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

#%% Build DNN
print('Building DNN')

# Initialize Data
model = GRU(input_size, hidden_size, output_size, num_layers, sequence_length, dropout).to(device)

# Loss Function
criterion = nn.MSELoss()

# Optimization Algorithm 
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,amsgrad=amsgrad)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,amsgrad=amsgrad)

# Learning Rate Scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, 
                                                 patience=3, cooldown=lr_step_size, 
                                                 min_lr=1e-5, verbose=True)

#%% Train the Network

h = model.init_hidden(batch_size = train_length)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = [] 

# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True, path='GRU_checkpoint.pt')

for epoch in range(num_epochs):
    
    ###################
    # Train the model #
    ###################
    model.train() # prep model for training
    h = model.init_hidden(batch_size = train_length)
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for (idx, batch) in loop: # in enumerate(train_loader)
    
        batch_x = batch[0].squeeze()
        batch_y = batch[1].squeeze()
    
        # print ("batch shape",batch_x.shape,batch_y.shape)
        
        # NOTE: make sure "out" in def forward(self, x, h0) in GRU class matches
        # with seq2one or seq2seq below
        
        # forward pass: compute predicted outputs by passing inputs to the model
        out,h = model(batch_x.float(),h)
        out = out[:,-1,0]
        
        # calculate the loss
        yTrue = batch_y.float() # seq2seq
        loss = criterion(out.squeeze(), yTrue) # if seq to one, only last timestep ([:,-1]) is used for loss 
        
        # record training loss
        train_losses.append(loss.item())
        
        # Backward
        optimizer.zero_grad() # clear the gradients of all optimized variables
        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # avoid gradient clipping
    
        # Gradient Descent or ADAM step
        optimizer.step()  # update the weights
        
        # detach hidden in between batches 
        h = h.detach()
        
        # Update Progress Bar
        loop.set_description(f"GRU Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
    ######################    
    # Validate the model #
    ######################
    model.eval() # prep model for evaluation
    h_val = model.init_hidden(batch_size = val_length)
    for (idx, batch) in enumerate(val_loader):
        
        batch_x = batch[0].squeeze()
        batch_y = batch[1].squeeze()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        out,h_val = model(batch_x.float(),h_val)
        out = out[:,-1,0]
        
        # calculate the loss
        yTrue = batch_y.float() # seq2seq
        loss = criterion(out.squeeze(), yTrue)
        
        # record validation loss
        valid_losses.append(loss.item())
    
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    epoch_len = len(str(num_epochs))
    print()
    print_msg = (f'[{epoch+1:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' + 
                 f'train_loss: {train_loss:.5f} ' + 
                 f'valid_loss: {valid_loss:.5f}')
    print(print_msg)    
    print()
    
    # Update the learning rate
    scheduler.step(valid_loss)
    
    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
    
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
# load the last checkpoint with the best model
model.load_state_dict(torch.load('GRU_checkpoint.pt'))

#%% Visualize Training and Validation Results

# Save Paths
timestr = time.strftime("%Y%m%d-%H%M%S")
cwd = os.getcwd()
directory = f'{timestr[:-2]}'
filename = 'loss_plot.png'
PATH = os.path.join(cwd,'GRU Results',directory,filename) 
os.makedirs(os.path.join(cwd, 'GRU Results',directory)) 

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('MSE loss')
# plt.ylim(0, 0.05) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.title('Minimum Validation Loss: {:.6f}\n'.format(np.min(avg_valid_losses)))
plt.tight_layout()
plt.show()
fig.savefig(PATH, bbox_inches='tight')

#%% Test the Trained Network

model.eval()
h = model.init_hidden(batch_size = test_length)

test_loss = 0.0
RMSE = []

target = []
pred = []

for (idx, batch) in enumerate(test_loader):
    
    batch_x = batch[0].squeeze()
    batch_y = batch[1].squeeze()
    
    # if len(batch_x) != num_trials:
    #     break
    
    # forward pass: compute predicted outputs by passing inputs to the model
    output,h = model(batch_x.float(),h)
    output = output[:,-1,0]
    # calculate the loss
    yPred = output.squeeze()
    yTrue = batch_y.float() # seq2one   
    loss = criterion(yPred, yTrue)
    # update test loss 
    test_loss += loss.item()*yPred.size(0)
    # compare predictions to true label
    rms = mean_squared_error(yTrue.detach().numpy(), yPred.detach().numpy(), squared=False)
    RMSE.append(rms)
    # for plotting
    target.append(yTrue.detach().numpy())
    pred.append(yPred.detach().numpy())
    
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
print('RMSE: {:.6f}\n'.format(np.average(RMSE)))

#%% Save Model

h = model.init_hidden(batch_size = 1)

filename = f'GRU NN {timestr[:-2]}.pt'
PATH = os.path.join(cwd,'GRU Results',directory,filename) 

# Save
net = model.state_dict()
torch.save(net, PATH)

#%% Visualize Testing Results

Targets = np.transpose(np.array(target))
Predictions = np.transpose(np.array(pred))
t = np.linspace(start=0,stop=len(Targets)/120,num=len(Targets))

## Plots
for i in range(0,num_trials):
    fig = plt.figure(figsize=(20,10))
    plt.plot(t , Targets[:,i], label='True')
    plt.plot(t , Predictions[:,i], label='Predicted')
    plt.xlabel('Time [s]')
    plt.ylabel('Ankle Torque [Nm]')
    plt.xlim(0,t[-1]) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.title(f'Test Trial {i+1}')
    plt.tight_layout()
    plt.show()
    filename = f'Test{i+1}.png'
    PATH = os.path.join(cwd,'GRU Results',directory,filename) 
    fig.savefig(PATH, bbox_inches='tight')

## Error Histogram
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

plt.xlabel('Error [Nm]')
plt.title('Testing Errors: ' + str(meanErr) + ' +/- ' + str(stdErr) + ' Nm')
plt.grid(True)
plt.tight_layout()
plt.show()
filename = 'Error Histogram.png'
PATH = os.path.join(cwd,'GRU Results',directory,filename) 
fig.savefig(PATH, bbox_inches='tight')

# Percent Error and Fit
percentError = mean_absolute_error(T, P)*100
percentFit = 100 - percentError

#%% Full Trials

train_iterator = iter(train_loader)
val_iterator = iter(val_loader)
test_iterator = iter(test_loader)
for i in range(num_trials):
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
      
    h_tr = model.init_hidden(batch_size = train_length)
    h_val = model.init_hidden(batch_size = val_length)
    h_test = model.init_hidden(batch_size = test_length)
    
    train_out, _ = model(train_x.squeeze().float(),h_tr)
    val_out, _ = model(val_x.squeeze().float(),h_val)
    test_out, _ = model(test_x.squeeze().float(),h_test)
    train_out = train_out[:,-1,0].detach().numpy()
    val_out = val_out[:,-1,0].detach().numpy()
    test_out = test_out[:,-1,0].detach().numpy()
    
    train_true = train_y.squeeze().float().detach().numpy()
    val_true = val_y.squeeze().float().detach().numpy()
    test_true = test_y.squeeze().float().detach().numpy()
    true_vals = np.concatenate((train_true, val_true, test_true), axis=0)
    
    time = np.linspace(start=30-len(true_vals)/120,stop=30,num=len(true_vals))
    time_tr = time[:train_length]
    time_val = time[train_length:train_length+val_length]
    time_test = time[train_length+val_length:]
    
    fig = plt.figure(figsize=(20,10))
    plt.plot(time , true_vals, color='black', label='True')
    plt.plot(time_tr , train_out, color='blue', label='Train')
    plt.plot(time_val , val_out, color='green', label='Validation')
    plt.plot(time_test , test_out, color='red', label='Test')
    plt.xlabel('Time [s]')
    plt.ylabel('Ankle Torque [Nm]')
    plt.title(f'Test Trial {i+1}')
    plt.xlim(0,30) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    filename = f'Trial{i+1}.png'
    PATH = os.path.join(cwd,'GRU Results',directory,filename) 
    fig.savefig(PATH, bbox_inches='tight')

#%% Convert Pytorch Model to Tensorflow using ONNX

# torch.manual_seed(1)

# with torch.no_grad():
#   # dummy_input = [torch.randn(1,input_size) for _ in range(sequence_length)] # make a sequence of length 5
#   # dummy_input = torch.cat(dummy_input).view(len(dummy_input), 1, -1)
  
#   dummy_batch_size = 1
#   dummy_input = torch.randn(dummy_batch_size, sequence_length, num_features, device=device) # (360,23,10,3)
#   h0 = model.init_hidden(batch_size = dummy_batch_size)
#   out, hn = model(dummy_input, h0, seqMode)

#   input_names = ['input', 'h0', 'seqMode']
#   output_names = ['output', 'hn']
    
#   torch.onnx.export(model, (dummy_input, h0, seqMode), 'GRU.onnx', 
#                     input_names=input_names, output_names=output_names)

# onnx_model = onnx.load("./GRU.onnx")
# filename = f'GRU NN {timestr[:-2]}.pb'
# PATH = os.path.join(cwd,'GRU Results',directory,filename)
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph(PATH)