"""

This file contains custom neural network class objects that can be called to
initialize a model or pass input information through the model to return an 
output prediction.
The purpose of these class objects is to contain model information and 
functionality to be passed to different training and evaluation scripts. 

DA-RNN Paper: https://arxiv.org/abs/1704.02971

"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # relu, tanh, etc.

# %% Feedforward Network


class FFN(nn.Module):
    def __init__(self, hyperparameters, constants):
        super(FFN, self).__init__()

        # Unpack relevant hyperparameters and dimensions
        self.dropout = nn.Dropout(hyperparameters['dropout'])
        self.sequence_length = hyperparameters['sequence length']
        self.input_size = constants['input size'] * self.sequence_length
        self.hidden_size = hyperparameters['hidden size']
        self.output_size = constants['output size']
        self.num_layers = hyperparameters['number of layers']
        self.batch_size = hyperparameters['batch size']

        # Activation function
        self.act = F.relu

        # Define first and last layers
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.final_layer = nn.Linear(self.hidden_size, self.output_size)

        # Use loop to define hidden layers
        self.hidden = []
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)

    def forward(self, x):
        # flatten features into single dimension
        y = torch.flatten(x, start_dim=1)
            
        # pass through first (input) layer of FFN
        y = self.dropout(self.act(self.input_layer(y)))

        # pass through hidden layers of FFN
        for i in range(self.num_layers):
            y = self.dropout(self.act(self.hidden[i](y)))

        # pass through final (output) layer of FFN
        out = self.final_layer(y)
        return out

# %% Gated Recurrent Unit


class GRU(nn.Module):

    def __init__(self, hyperparameters, constants):
        super(GRU, self).__init__()

        # Unpack relevant hyperparameters and dimensions
        self.input_size = constants['input size']
        self.output_size = constants['output size']
        self.hidden_size = hyperparameters['hidden size']
        self.num_layers = hyperparameters['number of layers']

        # Dropout layer not included on last GRU layer
        if self.num_layers > 1:
            self.gru = nn.GRU(self.input_size, self.hidden_size,
                              self.num_layers, batch_first=True,
                              dropout=hyperparameters['dropout'])
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size,
                              self.num_layers, batch_first=True)
        # Define final output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Define activation function
        self.relu = nn.ReLU()

        # Define computing device
        self.device = constants['device']

    def forward(self, x, h0):
        # forward pass through gru
        out, h0 = self.gru(x, h0)

        # forward pass through output layer and activation function
        out = self.fc(self.relu(out))
        return out, h0

    def init_hidden(self, batch_size):
        # retrieve first parameter in the gru model
        weight = next(self.parameters()).data
        # creates a hidden state tensor that has the same data type and device
        # as model parameter
        hidden = weight.new(self.num_layers, batch_size,
                            self.hidden_size).zero_().to(self.device)
        return hidden

# %% Dual-Stage Attention-Based Gated Recurrent Unit


class InputAttentionEncoder(nn.Module):
    def __init__(self, hyperparameters, constants, stateful=False):
        """
        :param: N: int
            number of time series
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with 
            values of the last cell state of previous time window or to 
            initialize it with zeros
        """
        super(self.__class__, self).__init__()

        # Unpack relevant hyperparameters and dimensions
        self.N = constants['input size']
        self.M = hyperparameters['hidden size']
        self.T = hyperparameters['sequence length']
        self.device = constants['device']
        self.encoder_gru = nn.GRUCell(input_size=self.N, hidden_size=self.M)

        # equation 8 matrices
        self.W_e = nn.Linear(self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)

    def forward(self, inputs):
        encoded_inputs = torch.zeros(
            (inputs.size(0), self.T, self.M)).to(self.device)

        # initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)

        for t in range(self.T):
            # concatenate hidden states
            h_c_concat = h_tm1

            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))

            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :]

            # calculate next hidden states (equation 11)
            h_tm1 = self.encoder_gru(weighted_inputs, h_tm1)

            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs


class TemporalAttentionDecoder(nn.Module):
    def __init__(self, hyperparameters, constants, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with 
            values of the last cell state of previous time window or to 
            initialize it with zeros
        """
        super(self.__class__, self).__init__()

        # Unpack relevant hyperparameters and dimensions
        self.M = hyperparameters['hidden size']
        self.P = hyperparameters['decoder size']
        self.T = hyperparameters['sequence length']
        self.stateful = stateful
        self.device = constants['device']
        self.decoder_gru = nn.GRUCell(input_size=1, hidden_size=self.P)

        # equation 12 matrices
        self.W_d = nn.Linear(self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias=False)

        # equation 15 matrix
        self.w_tilda = nn.Linear(self.M, 1)

        # equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)

    def forward(self, encoded_inputs):

        # initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(self.device)

        for t in range(self.T):
            # concatenate hidden states
            d_s_prime_concat = d_tm1

            # temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(
                1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)

            # normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)

            # create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)

            y_tilda_t = self.w_tilda(c_t)

            # calculate next hidden states (equation 16)
            d_tm1 = self.decoder_gru(y_tilda_t, d_tm1)

        # concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        # calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1


class DAGRU(nn.Module):   
    def __init__(self, hyperparameters, constants):
        super(self.__class__, self).__init__()
        self.device = constants['device']
        self.encoder = InputAttentionEncoder(hyperparameters, constants
                                             ).to(self.device)
        self.decoder = TemporalAttentionDecoder(hyperparameters, constants,
                                                ).to(self.device)

    def forward(self, X_history):
        out = self.decoder(self.encoder(X_history))
        return out
