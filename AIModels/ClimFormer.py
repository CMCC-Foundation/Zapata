'''
ClimFormer class, a subclass of InformerForPrediction
=====================================================

This class is a subclass of Informer
It contains classes for time series dataset, future time series dataset, and two subclasses of 
InformerForPrediction and TimeSeriesTransformerForPrediction


'''
import numpy as np
import xarray as xr
import pandas as pd

from typing import Optional, List, Union, Tuple

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


class TimeSeriesDataset(Dataset):
    '''
    Class for time series dataset.
    Includes time feature for transformers

    PARAMETERS
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
        If not  None  contain Time Features
    
        
    ATTRIBUTES
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

class TimeSeriesFuture(Dataset):
    '''
    Class for time series dataset.
    Includes future time feature for prediction with informer

    PARAMETERS
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
        If not  None  contain Time Features
    shift: 
        Overlap between source and target, for trasnformers
         overlap = 0  for LSTM  overlap  should be TIN-T
        
    ATTRIBUTES
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

    def __init__(self, datasrc, datatgt, TIN, MIN, T, K, Tpredict, time_features=None):
        self.datasrc = datasrc
        self.datatgt = datatgt
        self.TIN = TIN
        self.MIN = MIN
        self.T = T
        self.Tpredict = Tpredict
        self.K = K
        self.time_features = time_features
      
        
        
    def __len__(self):
        return len(self.datasrc) - self.TIN - self.Tpredict + 1
    
    def __getitem__(self, idx):
        input_seq = self.datasrc[idx:idx+self.TIN, :self.MIN]
        target_seq = self.datatgt[idx+self.TIN:idx+self.TIN+self.T, :self.K]  
        pasft = self.time_features[idx:idx+self.TIN,:]
        futft = self.time_features[idx+self.TIN:idx+self.TIN+self.Tpredict,:]
       
        return input_seq, target_seq, pasft, futft


class ClimFormer(tr.InformerForPrediction):
    '''
    Class for training and prediction with InformerForPrediction model
    from ``transformers`` library
    '''
    def __init__(self, config):
        super().__init__(config)
        tr.InformerForPrediction(config)
    
class TrasFormer(tr.TimeSeriesTransformerForPrediction):
    '''
    Class for training and prediction with TimeSeriesTransformerForPrediction model
    from ``transformers`` library
    '''
    def __init__(self, *args):
        tr.TimeSeriesTransformerForPrediction.__init__(self, *args)
