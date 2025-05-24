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
import os,sys
import math
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
from sklearn.base import BaseEstimator, TransformerMixin

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
    def __init__(self, name, levels,area, mr, dstart='1/1/1940', dtend='12/31/2022',dropX=False):
        self.name = name
        self.levels = levels
        self.mr = mr
        self.area = area
        self.dstart = dstart
        self.dtend = dtend
        self.dropX = dropX
        
    def __call__(self):
        return self.name,self.levels
    def __repr__(self):
        '''  Printing Information '''
        print(f'Field: {self.name}, Levels: {self.levels}, Area: {self.area}, DropX: {self.dropX}')
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

class PositionalEmbedding(nn.Module):
    def __init__(self, T, neof, embedding_dim):
        """
        Initialize positional embedding module.
        
        Parameters:
        - T (int): Maximum sequence length (time dimension).
        - neof (int): Feature space dimension of input.
        - embedding_dim (int): Dimensionality of the positional embedding for each axis.
        """
        super(PositionalEmbedding, self).__init__()
        
        # Learnable embeddings for temporal and feature positions
        self.time_embedding = nn.Embedding(T, embedding_dim)
        self.feature_embedding = nn.Embedding(neof, embedding_dim)
        
        # Linear layer projecting embedding_dim to 1 scalar per position
        self.linear = nn.Linear(embedding_dim, 1)
        
    def forward(self, X):
        """
        Apply positional embeddings to the input sequence.
        
        Parameters:
        - X (Tensor): Input tensor of shape (batch_size, T, neof)
        
        Returns:
        - Tensor: Positional embedding applied tensor of shape (batch_size, T, neof)
        """
        batch_size, T, neof = X.size()
        
        # Create time and feature position indices
        time_positions = torch.arange(T, device=X.device).unsqueeze(0).repeat(batch_size, 1)  
          # (batch_size, T)
        feature_positions = torch.arange(neof, device=X.device).unsqueeze(0).repeat(batch_size, 1)  
          # (batch_size, neof)
        
        # Get embeddings for time and feature positions
        time_pos_emb = self.time_embedding(time_positions)  # (batch_size, T, embedding_dim)
        feature_pos_emb = self.feature_embedding(feature_positions)  # (batch_size, neof, embedding_dim)
        
        # Combine positional embeddings across both axes
        # Expand dims so we can broadcast: (T, 1) + (1, neof) => (T, neof)
        time_pos_emb = time_pos_emb.unsqueeze(-2)     # (batch_size, T, 1, embedding_dim)
        feature_pos_emb = feature_pos_emb.unsqueeze(-3)  # (batch_size, 1, neof, embedding_dim)
        
        combined_pos_emb = time_pos_emb + feature_pos_emb  # (batch_size, T, neof, embedding_dim)
        
        # Project embedding_dim -> 1, then remove that last dimension
        combined_pos_emb = self.linear(combined_pos_emb).squeeze(-1)  # (batch_size, T, neof)
        
        # Add positional embeddings to input
        X_with_pos = X + combined_pos_emb  # (batch_size, T, neof)
        
        return X_with_pos

class FixedPositionalEmbedding(nn.Module):
    """
    Creates fixed (non-learnable) positional embeddings for
    both the time dimension (T) and feature dimension (neof).
    """
    def __init__(self, max_T, max_neof, embedding_dim=64):
        super().__init__()
        
        # 1) Create zero tensors for time and feature embeddings
        time_pe = torch.zeros(max_T, embedding_dim)      # (max_T, embedding_dim)
        feature_pe = torch.zeros(max_neof, embedding_dim)  # (max_neof, embedding_dim)
        
        # 2) Prepare position indices for time and features
        position_t = torch.arange(0, max_T, dtype=torch.float).unsqueeze(1)  # (max_T, 1)
        position_f = torch.arange(0, max_neof, dtype=torch.float).unsqueeze(1)  # (max_neof, 1)
        
        # 3) Compute the divisor terms for the sine-cosine formula
        #    The standard Transformer approach uses exponentials of the form:
        #      exp( - (2*i)/embedding_dim * ln(10000) )
        #    Here, we do a similar but slightly different approach (still valid).
        div_term_t = torch.exp(
            torch.arange(0, embedding_dim, 2).float() 
            * (-math.log(10000.0) / embedding_dim)
        )
        div_term_f = torch.exp(
            torch.arange(0, embedding_dim, 2).float() 
            * (-math.log(10000.0) / embedding_dim)
        )
        
        # 4) Fill time_pe with sine on even dims, cos on odd dims
        #    shape: (max_T, embedding_dim)
        #    position_t * div_term_t -> broadcast: (max_T, 1) * (num_even_dims) => (max_T, num_even_dims)
        time_pe[:, 0::2] = torch.sin(position_t * div_term_t)  # Even indices
        time_pe[:, 1::2] = torch.cos(position_t * div_term_t)  # Odd indices
        
        # 5) Fill feature_pe with sine on even dims, cos on odd dims
        feature_pe[:, 0::2] = torch.sin(position_f * div_term_f)
        feature_pe[:, 1::2] = torch.cos(position_f * div_term_f)
        
        # 6) Register them as buffers => not learnable but move with .to(device)
        self.register_buffer("time_pe", time_pe)       # (max_T, embedding_dim)
        self.register_buffer("feature_pe", feature_pe) # (max_neof, embedding_dim)
        
        # 7) Linear layer to map from embedding_dim -> 1 offset per (t,feature)
        #    (If you wanted an offset of size 'neof', you'd do nn.Linear(embedding_dim, neof).)
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, X):
        """
        X: shape (batch_size, T, neof)
        
        Returns: shape (batch_size, T, neof)
                 with a fixed (non-learnable) sinusoidal offset added.
        """
        batch_size, T, neof = X.shape
        
        # (A) Slice the needed portion of time_pe: (T, embedding_dim) -> (1, T, 1, embedding_dim)
        time_pos_emb = self.time_pe[:T].unsqueeze(0).unsqueeze(2)
        
        # (B) Slice the needed portion of feature_pe: (neof, embedding_dim) -> (1, 1, neof, embedding_dim)
        feature_pos_emb = self.feature_pe[:neof].unsqueeze(0).unsqueeze(1)
        
        # (C) Combine via addition, broadcasting: (1, T, neof, embedding_dim)
        combined_pos_emb = time_pos_emb + feature_pos_emb
        
        # (D) Map embedding_dim -> 1 to get a scalar offset per (t, feature): shape => (1, T, neof, 1) -> squeeze => (1, T, neof)
        combined_pos_emb = self.linear(combined_pos_emb).squeeze(-1)
        
        # (E) Broadcast over batch dimension: (batch_size, T, neof)
        X_with_pos = X + combined_pos_emb
        return X_with_pos


class FeaturePositionalEmbedding(nn.Module):
    """
    Creates fixed (non-learnable) positional embeddings for the feature dimension only.
    
    Given an input (batch_size, T, neof), we produce a feature-wise offset
    of shape (1, 1, neof), then broadcast-add it to the input.
    """
    def __init__(self, max_neof, embedding_dim=64):
        """
        Args:
            max_neof (int): maximum feature dimension for which to create embeddings
            embedding_dim (int): dimensionality of the sine-cosine embedding
        """
        super().__init__()
        
        # Precompute sine-cosine embeddings for [0..max_neof - 1]
        feature_pe = torch.zeros(max_neof, embedding_dim)  # (max_neof, embedding_dim)
        position_f = torch.arange(0, max_neof, dtype=torch.float).unsqueeze(1)  # (max_neof, 1)
        
        # Exponential spacing of frequencies, as in the Transformer paper
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        
        # Apply sine to even indices, cosine to odd indices
        feature_pe[:, 0::2] = torch.sin(position_f * div_term)
        feature_pe[:, 1::2] = torch.cos(position_f * div_term)
        
        # Register as a buffer (not a parameter)
        self.register_buffer("feature_pe", feature_pe)  # (max_neof, embedding_dim)
        
        # Linear layer to map embedding_dim -> 1 scalar per feature position
        # If you prefer a neof-dimensional offset, use nn.Linear(embedding_dim, neof).
        self.linear = nn.Linear(embedding_dim, 1)
    
    def forward(self, X):
        """
        Args:
            X: Tensor of shape (batch_size, T, neof)
        
        Returns:
            Tensor of the same shape, with a feature-wise positional offset added.
        """
        batch_size, T, neof = X.shape
        
        # Slice the needed portion of feature_pe: shape => (neof, embedding_dim)
        # Then reshape to (1, 1, neof, embedding_dim) for broadcast
        feature_pos_emb = self.feature_pe[:neof].unsqueeze(0).unsqueeze(0)  
          # (1, 1, neof, embedding_dim)
        
        # Map (embedding_dim) -> 1 scalar per feature
        # Result shape => (1, 1, neof, 1) => squeeze => (1, 1, neof)
        feature_pos_emb = self.linear(feature_pos_emb).squeeze(-1)  
          # (1, 1, neof)
        
        # Broadcast-add to X, which is (batch_size, T, neof)
        # feature_pos_emb expands over batch_size and T
        X_with_pos = X + feature_pos_emb
        
        return X_with_pos



class SymmetricFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scales each feature to a symmetric range [-scale, +scale] around zero,
    following the scikit-learn transformer API.

    For feature i, the data's minimum is mapped to -feature_scales[i],
    and the data's maximum is mapped to +feature_scales[i].
    
    Parameters
    ----------
    feature_scales : array-like of shape (n_features,), optional (default=None)
        If None, all features are scaled to [-1, +1].
        Otherwise, each feature i is scaled to [-feature_scales[i], +feature_scales[i]].

    Attributes
    ----------
    data_min_ : ndarray of shape (n_features,)
        Per-feature minimum seen in the data during fit.
    data_max_ : ndarray of shape (n_features,)
        Per-feature maximum seen in the data during fit.
    n_features_ : int
        Number of features in the fitted data.
    feature_scales_ : ndarray of shape (n_features,)
        Final validated array of per-feature scales.

    Example
    -------
    >>> import numpy as np
    >>> X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
    >>> # Suppose we want feature 0 scaled to [-1, +1] and feature 1 to [-5, +5]
    >>> feature_scales = [1.0, 5.0]
    >>> scaler = SymmetricFeatureScaler(feature_scales=feature_scales)
    >>> scaler.fit(X)
    SymmetricFeatureScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> X_scaled
    array([[-1.        , -5.        ],
           [ 0.        ,  0.        ],
           [ 1.        ,  5.        ]])
    >>> X_orig = scaler.inverse_transform(X_scaled)
    >>> X_orig
    array([[ 1., 10.],
           [ 2., 20.],
           [ 3., 30.]])
    """

    def __init__(self, feature_scales=None):
        self.feature_scales = feature_scales

    def fit(self, X, y=None):
        """Learn the per-feature min and max from the training data."""
        X = self._check_array(X)
        self.n_features_ = X.shape[1]

        # Compute data min and max per feature
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)

        # If feature_scales is None, default each feature to 1.0 => [-1, +1]
        if self.feature_scales is None:
            self.feature_scales_ = np.ones(self.n_features_, dtype=float)
        else:
            self.feature_scales_ = np.array(self.feature_scales, dtype=float)
            if self.feature_scales_.shape != (self.n_features_,):
                raise ValueError(
                    f"`feature_scales` must be of shape (n_features,), "
                    f"but got {self.feature_scales_.shape}."
                )

        return self

    def transform(self, X):
        """Scale the input data X to [-feature_scales[i], +feature_scales[i]] per feature."""
        X = self._check_array(X)
        self._check_is_fitted()

        # Map [data_min_, data_max_] to [0, 1]
        denominator = self.data_max_ - self.data_min_
        # Avoid division by zero for constant features
        denominator = np.where(denominator == 0, 1, denominator)

        X_std = (X - self.data_min_) / denominator  # in [0, 1]

        # Map [0, 1] to [-1, +1]
        X_sym = 2 * X_std - 1  # in [-1, +1]

        # Map [-1, +1] to [-scale, +scale]
        X_scaled = X_sym * self.feature_scales_

        return X_scaled

    def inverse_transform(self, X):
        """
        Revert data back to original range:
        [-scale, +scale] => [-1, +1] => [0, 1] => [data_min_, data_max_].
        """
        X = self._check_array(X)
        self._check_is_fitted()

        denominator = self.data_max_ - self.data_min_
        denominator = np.where(denominator == 0, 1, denominator)

        # First, map [-scale, +scale] back to [-1, +1]
        X_sym = X / self.feature_scales_

        # Then map [-1, +1] to [0, 1]
        X_std = (X_sym + 1) / 2

        # Finally map [0, 1] to [data_min_, data_max_]
        X_orig = X_std * denominator + self.data_min_

        return X_orig

    def _check_array(self, X):
        """Ensure X is at least 2D."""
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _check_is_fitted(self):
        """Verify that fit has been called by checking learned attributes."""
        if not hasattr(self, 'data_min_') or not hasattr(self, 'data_max_'):
            raise AttributeError(
                "This SymmetricFeatureScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
class IdentityScaler(BaseEstimator, TransformerMixin):
    """
    A no-op (identity) scaler that complies with scikit-learn's
    estimator API. It leaves data unchanged.
    """
    def fit(self, X, y=None):
        # Nothing to compute
        return self

    def transform(self, X, y=None):
        # Return the input as-is
        return X

    def inverse_transform(self, X, y=None):
        # Return the input as-is
        return X