'''
ClimFormer class, a subclass of InformerForPrediction
=====================================================

This class is a subclass of Informer
It contains classes for time series dataset, future time series dataset, and two subclasses of 
InformerForPrediction and TimeSeriesTransformerForPrediction

Utilities 
---------

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
        If not  None  contain Time Features
    
        
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

class TimeSeriesFuture(Dataset):
    '''
    Class for time series dataset.
    Includes future time feature for prediction with informer

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
        If not  None  contain Time Features
    shift: 
        Overlap between source and target, for trasnformers
         overlap = 0  for LSTM  overlap  should be TIN-T
        
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
    '''
    def __init__(self, config):
        super().__init__(config)
        tr.InformerForPrediction(config)
    
class TrasFormer(tr.TimeSeriesTransformerForPrediction):
    '''
    Class for training and prediction with TimeSeriesTransformerForPrediction model
    '''
    def __init__(self, *args):
        tr.TimeSeriesTransformerForPrediction.__init__(self, *args)

class ClimFormerDeter(tr.InformerModel):
    '''
    Class for training and prediction with InformerModel model,
    deterministic version
    '''
    def __init__(self, config):
        super().__init__(config)
        self.model = tr.InformerModel(config)
        self.projection = nn.Linear(self.config.d_model, self.config.output_size) 
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        
        outputs = self.model(
             past_values=past_values,
             past_time_features=past_time_features,
             past_observed_mask=past_observed_mask,
             static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features)

        return self.projection(outputs.last_hidden_state), outputs
