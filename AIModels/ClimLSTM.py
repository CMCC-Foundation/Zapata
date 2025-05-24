'''
`ClimLSTM` class for training, validation and prediction of a LSTM model for climate data.

Uses pytorch model libraries
'''
import numpy as np
import xarray as xr
import pandas as pd

import scipy.linalg as sc

import matplotlib.pyplot as plt
# import datetime
import time as tm
from alive_progress import alive_bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import transformers as tr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import zapata.computation as zcom
import zapata.data as zd
import zapata.lib as zlib
import zapata.mapping as zmap
import zapata.koopman as zkop

class TimeSeriesDataset(Dataset):
    '''
    Prepare Time Series data set for suitable for the pytorch Dataloader
    Set up for LSTM

    PARAMETERS
    ==========
    
    data:
        Data to be input
    TIN:
        input sequence length
    MIN:
        Input number of features
    T:
        Prediction sequence length
    K:
        Output Number of features
    S:
        Shift between input and target sequence

    Note
    ====
        A rigid shift in the data is produced to train on `T` time steps
        $x_1, x_2, \ldots, x_{n-1}$ and the target is $x_2, x_3, \ldots, x_{n}$
        The size is already the number of expected forecasts step
    '''
    def __init__(self, data, TIN, MIN, T, K):
        self.data = data
        self.TIN = TIN
        self.MIN = MIN
        self.T = T
        self.K = K 
        
    def __len__(self):
        return len(self.data) - self.TIN - self.T +1
        
    
    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.TIN, :self.MIN]
        target_seq = self.data[idx+self.T:idx+self.TIN+self.T, :self.K]
        return input_seq, target_seq
    

class ClimLSTM(nn.Module):
    '''
    LSTM model for time series
    Based on pytorch
    
    '''
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, num_classes) # fully connected 
        # self.fc_2 =  nn.Linear(input_size, input_size)
        self.relu = nn.GELU()
        # self.dropout = nn.Dropout(0.1)
        
    def forward(self,x):
        
        # propagate input through LSTM
        # xx = self.fc_2(x)
        # nb,_,_ = x.shape
        # h0 = torch.randn(1, nb, self.hidden_size,dtype=torch.float64) # hidden state
        # c0 = torch.randn(1, nb, self.hidden_size,dtype=torch.float64) # cell state
        output, _ = self.lstm(x) 
        out = self.fc_1(output) # first dense
        
        return out
    
    def train_model(self, iterator, optimizer, criterion, scheduler, clip, device):
    
       
        self.train()
        epoch_loss = 0
    
        for i, (x,y) in enumerate(iterator):  
            src = x.to(device) #batch.src
            trg = y.to(device) #batch.trg  
            output = self(src)   
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            
            optimizer.zero_grad()
            loss = criterion(output, trg)
            # print(loss)
        
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            
            epoch_loss += loss.item()

        scheduler.step(loss)    

        return epoch_loss / len(iterator)

    def evaluate_model(self, iterator, criterion,device):
        
        
        self.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for i, (x,y) in enumerate(iterator):

                src = x.to(device) #batch.src
                trg = y.to(device) #batch.trg

                output = self(src) #turn off teacher forcing

                
                loss = criterion(output, trg)
                
                epoch_loss += loss.item()
        
        
        return epoch_loss / len(iterator)

    def predict(self, iterator, K, Tpredict,device):
        
        self.eval()
        
        with torch.no_grad():
            
            for i, (x,y) in enumerate(iterator):
                
                src = x.to(device) #batch.src
                # trg = y #batch.trg
                batch, L, Nfeatures = src.shape
                preds = torch.zeros(batch, Tpredict,K,device=device)

                for it in range(Tpredict):
                    output = self(src) #turn off teacher forcing
                    preds[:,it,:] = output[:,-1,:]
                    src = output
                   
            
                temp = preds.cpu().detach().numpy()
                
                if i == 0:
                    trainpl = temp.copy()
                else:
                    trainpl = np.concatenate((trainpl, temp))      
            

        return trainpl
