"""
Created on Wed Apr 21 12:32:47 2021

Nueral Network Model Class Definitions

@author: cpras
"""

#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F  # relu, tanh, etc.
import math

#%% Feedforward NN (FFN)

class FFN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers, sequence_length, dropout):
        super(FFN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.sequence_length = sequence_length
        self.input_size = input_size * self.sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.act = F.relu
        self.final_layer = nn.Linear(self.hidden_size, self.output_size)
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden = []
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size,self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        
    def forward(self,x):
        y = torch.flatten(x,start_dim=1)
        y = self.dropout(self.act(self.input_layer(y)))
        for i in range(self.num_layers):
            y = self.dropout(self.act(self.hidden[i](y)))
        out = self.final_layer(y)
        return out

    # def __init__(self, input_size, hidden_size, output_size, dropout): 
    #     super(FFN, self).__init__()
    #     self.fc1 = nn.Linear(input_size, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)
    #     self.out = nn.Linear(hidden_size, output_size)
    #     self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     x = torch.flatten(x,start_dim=1)
    #     y = self.dropout(F.relu(self.fc1(x)))
    #     y = self.dropout(F.relu(self.fc2(y)))
    #     y = self.out(y)  # no activation
    #     return y

#%% Gated Recurrent Unit NN (GRU)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, sequence_length, dropout, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.num_layers > 1:
            self.gru = nn.GRU(input_size, hidden_size, 
                              num_layers, batch_first=True, dropout=dropout)
        else:
            self.gru = nn.GRU(input_size, hidden_size, 
                              num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.device = device
     
    def forward(self, x, h0):
        out, h0 = self.gru(x, h0)
        out = self.fc(self.relu(out))
        return out, h0
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
        return hidden

#%% Transformer Network (Transformer)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        x + self.pe[:x.size(0), :]
        return self.dropout(x)
       

class TransformerModel(nn.Module):
    def __init__(self,nout,ninp,nhead,nhid,num_layers,dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer' 
        self.src_mask = None
        
        self.pos_encoder = PositionalEncoding(ninp,dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, 
                                                        dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.encoder = nn.Embedding(nout,ninp) # added
        self.ninp = ninp
        self.decoder = nn.Linear(ninp,nout)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.encoder.weight.data.uniform_(-initrange, initrange) # added
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        # src = self.encoder(src) * math.sqrt(self.ninp) # added
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output) #seq2seq
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

#%% Dual-Stage Attention-Based Gated Recurrent Unit (DA-RNN)

class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, device, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        self.device = device
        
        # self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)
        self.encoder_gru = nn.GRUCell(input_size=self.N, hidden_size=self.M)
        
        #equation 8 matrices
        # self.W_e = nn.Linear(2*self.M, self.T)
        self.W_e = nn.Linear(self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)
    
    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).to(self.device)
        
        #initiale hidden states
        # s_tm1 = torch.zeros((inputs.size(0), self.M)).to(device)
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(self.device)
        
        for t in range(self.T):
            #concatenate hidden states
            # h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            h_c_concat = h_tm1
            
            #attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            # y = self.U_e(inputs.permute(0, 1, 3, 2)) # CHANGED
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))
        
            #normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)
            
            #weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
            # calculate next hidden states (equation 11)
            # h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            h_tm1 = self.encoder_gru(weighted_inputs, h_tm1)
            
            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs   
    
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, device, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful
        self.device = device
        
        # self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)
        self.decoder_gru = nn.GRUCell(input_size=1, hidden_size=self.P)
        
        #equation 12 matrices
        self.W_d = nn.Linear(self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.M, 1)
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)
        
    def forward(self, encoded_inputs):
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(self.device)
        # s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(device)
        
        for t in range(self.T):
            #concatenate hidden states
            # d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            d_s_prime_concat = d_tm1
            
            #print(d_s_prime_concat)
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            
            # #concatenate c_t and y_t
            # y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            # #create y_tilda
            # y_tilda_t = self.w_tilda(y_c_concat)
            
            y_tilda_t = self.w_tilda(c_t)
            
            #calculate next hidden states (equation 16)
            # d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
            d_tm1 = self.decoder_gru(y_tilda_t, d_tm1)
        
        #concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1
    
class DARNN(nn.Module):
    def __init__(self, N, M, P, T, device, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.device = device
        self.encoder = InputAttentionEncoder(N, M, T, device, stateful_encoder).to(self.device)
        self.decoder = TemporalAttentionDecoder(M, P, T, device, stateful_decoder).to(self.device)
    def forward(self, X_history):
        out = self.decoder(self.encoder(X_history))
        return out

