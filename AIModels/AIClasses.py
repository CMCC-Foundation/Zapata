'''
Classes for the AI project

Classes
=======

Field: 
    Class for fields

EarlyStopping: 
    Class for early stopping

TimeSeriesDataset: 
    Class for time series dataset

'''
# import os,sys
# import math
import numpy as np
import xarray as xr
import pandas as pd

import scipy.linalg as sc

import matplotlib.pyplot as plt
# import datetime
import time as tm

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

class Field():
    '''
    Class for fields
    
    Parameters
    ==========
    
    name: string
        Name of the field
    levels: string
        Level of the field
    area: string
        Area to be analyzed, possible values are
        
        * 'TROPIC': Tropics
        * 'GLOBAL': Global
        * 'PACTROPIC': Pacific Tropics
        * 'WORLD': World
        * 'EUROPE': Europe
        * 'NORTH_AMERICA': North America
        * 'NH-ML': Northern Hemisphere Mid-Latitudes

    mr: float
        Number of EOF retained
    dstart: string
        Start date for field
    dtend: string
        End date for field
        
    Attributes
    ==========

    name: string
        Name of the field
    levels: string
        Level of the field
    area: string
        Area of the field
    mr: float
        Number of EOF retained
    dstart: string
        Start date for field
    dtend: string
        End date for field
    
    '''
    def __init__(self, name, levels,area, mr, dstart='1/1/1940', dtend='12/31/2022'):
        self.name = name
        self.levels = levels
        self.mr = mr
        self.area = area
        self.dstart = dstart
        self.dtend = dtend
        
    def __call__(self):
        return self.name,self.levels
    def __repr__(self):
        '''  Printing Information '''
        print(f'Field: {self.name}, Levels: {self.levels}, Area: {self.area}')
        print(f'EOF Retained: {self.mr}, StartDate: {self.dstart}, EndDate: {self.dtend}')
        return '\n'
    
    # Early stopping class
class EarlyStopping:
    '''
    Class for early stopping
    
    Parameters
    ==========
    
    patience: int
        Number of epochs to wait before stopping
    verbose: boolean
        If True, print the epoch when stopping
    delta: float
        Minimum change in loss to be considered an improvement
        
    Attributes
    ==========
        
    patience: int
        Number of epochs to wait before stopping
    verbose: boolean
        If True, print the epoch when stopping
    delta: float
        Minimum change in loss to be considered an improvement
    counter: int
        Number of epochs since last improvement
    best_score: float
        Best loss score
    early_stop: boolean
        If True, stop the training
        
        '''
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop
    
class TimeSeriesDataset(Dataset):
    '''
    Class for time series dataset.
    Includes time feature for transformers

    Parameters
    ==========

    datasrc: numpy array
        Source data
    datatgt: numpy array
        Target data
    TIN: int
        Input time steps
    MIN: int
        Input variables size
    T: int
        Predictions time steps
    K: int
        Output variables size 
    time_features: numpy array (optional)
        If not `None` contain Time Features
    shift: 
        Overlap between source and target, for trasnformers
        `overlap = 0` for LSTM `overlap` should be TIN-T
        
    Attributes
    ==========

    datasrc: numpy array
        Source data
    datatgt: numpy array
        Target data
    time_features: numpy array
        Time features
    TIN: int
        Input time steps
    MIN: int
        Input variables
    T: int
        Output time steps
    K: int
        Output variables

    '''

    def __init__(self, datasrc, datatgt, TIN, MIN, T, K, time_features=None):
        self.datasrc = datasrc
        self.datatgt = datatgt
        self.TIN = TIN
        self.MIN = MIN
        self.T = T
        self.K = K
        self.time_features = time_features
      
        
        
    def __len__(self):
        return len(self.datasrc) - self.TIN - self.T + 1
    
    def __getitem__(self, idx):
        input_seq = self.datasrc[idx:idx+self.TIN, :self.MIN]
        target_seq = self.datatgt[idx+self.TIN:idx+self.TIN+self.T, :self.K]  
        pasft = self.time_features[idx:idx+self.TIN,:]
        futft = self.time_features[idx+self.TIN:idx+self.TIN+self.T,:]
       
        return input_seq, target_seq, pasft, futft

