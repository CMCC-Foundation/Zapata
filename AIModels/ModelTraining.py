'''
Training,validation and prediction methods for the Informer model.
==================================================================
A thin wrapper around the InformerForPrediction class, with additional methods for training, validation and prediction.

Utilities 
---------

'''

from typing import List, Optional, Tuple, Union

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

from transformers.modeling_outputs import Seq2SeqTSModelOutput, SampleTSPredictionOutput,Seq2SeqTSPredictionOutput

   
def train_model(model, epoch, train_loader, optimizer, lr=0.001, patience=5,clip=1.0,device=None,criterion=None):
    model.train()
    train_loss = 0.0
    T = model.config.prediction_length
    TIN = model.config.context_length + max( model.config.lags_sequence)
    MIN = model.config.input_size
    K = model.config.input_size

    for src, tgt, pasft, futft  in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        pasft, futft = pasft.to(device), futft.to(device)
        
        batch_size,_,_ = src.shape
        optimizer.zero_grad()
        # Create past and future time features
        # print(src.shape,tgt.shape,pasft.shape,futft.shape)
        pasobs = torch.ones([batch_size,TIN,MIN],dtype=torch.float32, device=device)
        

        optimizer.zero_grad()
        if criterion is None:
            output = model(
                past_values=src,
                past_time_features=pasft,
                past_observed_mask=pasobs,
                # static_categorical_features=batch["static_categorical_features"],
                # static_real_features=batch["static_real_features"],
                future_values=tgt,
                future_time_features=futft,
                )
            loss = output.loss
        else:
            out, _ = model(
                past_values=src,
                past_time_features=pasft,
                past_observed_mask=pasobs,
                # static_categorical_features=batch["static_categorical_features"],
                # static_real_features=batch["static_real_features"],
                future_values=tgt,
                future_time_features=futft,
                )
            loss = criterion(out, tgt)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        
    train_loss /= len(train_loader)


    return model, train_loss

def validate_model(model, epoch, val_loader, lr=0.001, patience=5,clip=1.0,device=None,criterion=None):

    # Validation
    model.eval()
    val_loss = 0.0
    TIN = model.config.context_length + max( model.config.lags_sequence)
    MIN = model.config.input_size
    K = model.config.input_size
    T = model.config.prediction_length
    with torch.no_grad():
        for src, tgt, pasft, futft in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            pasft, futft = pasft.to(device), futft.to(device)
            
            batch_size,_,_ = src.shape
            
            pasobs = torch.ones([batch_size,TIN,MIN],dtype=torch.float32, device=device)
            
            # during inference, one only provides past values
            # as well as possible additional features
            # the model autoregressively generates future values
            if criterion is None:
                output = model(
                    past_values=src,
                    past_time_features=pasft,
                    past_observed_mask=pasobs,
                    # static_categorical_features=batch["static_categorical_features"],
                    # static_real_features=batch["static_real_features"],
                    future_values=tgt,
                    future_time_features=futft,
                    )
                loss = output.loss
            else:
                out, _ = model(
                    past_values=src,
                    past_time_features=pasft,
                    past_observed_mask=pasobs,
                    # static_categorical_features=batch["static_categorical_features"],
                    # static_real_features=batch["static_real_features"],
                    future_values=tgt,
                    future_time_features=futft,
                    )
                loss = criterion(out, tgt)
        
            # loss.backward()
            val_loss += loss.item()
            
            # print('val',val_loss)

            val_loss /= len(val_loader)           

    return model,val_loss

def predict(model,  val_loader, Tpredict, device=None,criterion=None):
    # model.to(device)
    model.eval()
    TIN = model.config.context_length + max( model.config.lags_sequence)
    MIN = model.config.input_size
    with alive_bar(len(val_loader),force_tty=True) as bar:
        for i, (src, tgt, pasft, futft) in enumerate(val_loader):
            tm.sleep(0.005)
            
            src, tgt = src.to(device), tgt.to(device)
            pasft, futft = pasft.to(device), futft.to(device)

            batch_size,_,_ = src.shape
            
            pasobs = torch.ones([batch_size,TIN,MIN],dtype=torch.float32, device=device)
            
            # during inference, one only provides past values
            # as well as possible additional features
            # the model autoregressively generates future values
        
            if criterion is None:
                output = model.generate(
                past_values=src,
                past_time_features=pasft,
                past_observed_mask=pasobs,
                future_time_features=futft)
            else:
                output = deter_generate(model,
                past_values=src,
                past_time_features=pasft,
                past_observed_mask=pasobs,
                future_time_features=futft)

            if i == 0 :
                temp = output['sequences']
            else:
                temp = torch.cat([temp, output['sequences']])
                       
            bar()
            
    return temp
def deter_generate(
        model,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SampleTSPredictionOutput:
        r"""
        Greedily generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
                Past values of the time series, that serve as context in order to predict the future. The sequence size
                of this tensor must be larger than the `context_length` of the model, since the model will use the
                larger size to construct lag features, i.e. additional values from the past which are added in order to
                serve as "extra context".

                The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if
                no `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
                look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length
                of the past.

                The `past_values` is what the Transformer encoder gets as input (with optional additional features,
                such as `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

                Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

                For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number
                of variates in the time series per time step.
            past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
                Required time features, which the model internally will add to `past_values`. These could be things
                like "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features).
                These could also be so-called "age" features, which basically help the model know "at which point in
                life" a time-series is. Age features have small values for distant past time steps and increase
                monotonically the more we approach the current time step. Holiday features are also a good example of
                time features.

                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,
                where the position encodings are learned from scratch internally as parameters of the model, the Time
                Series Transformer requires to provide additional time features. The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
            future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
                Required time features for the prediction window, which the model internally will add to sampled
                predictions. These could be things like "month of year", "day of the month", etc. encoded as vectors
                (for instance as Fourier features). These could also be so-called "age" features, which basically help
                the model know "at which point in life" a time-series is. Age features have small values for distant
                past time steps and increase monotonically the more we approach the current time step. Holiday features
                are also a good example of time features.

                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,
                where the position encodings are learned from scratch internally as parameters of the model, the Time
                Series Transformer requires to provide additional time features. The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
                Optional static categorical features for which the model will learn an embedding, which it will add to
                the values of the time series.

                Static categorical features are features which have the same value for all time steps (static over
                time).

                A typical example of a static categorical feature is a time series ID.
            static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
                Optional static real features which the model will add to the values of the time series.

                Static real features are features which have the same value for all time steps (static over time).

                A typical example of a static real feature is promotion information.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Return:
            [`SampleTSPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, prediction_length)` or `(batch_size, number of samples, prediction_length, input_size)` for
            multivariate predictions.
        """
        out, outputs = model(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=True,
        )

        decoder = model.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = 1
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        future_samples = []

        # greedy decoding
        for k in range(model.config.prediction_length):
            lagged_sequence = model.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_output = decoder(inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden)
            dec_last_hidden = dec_output.last_hidden_state[:,0,:].unsqueeze(1)
            # print(f'{dec_last_hidden.shape}')
            # params = model.model.parameter_projection(dec_last_hidden[:, -1:])
            # distr = model.model.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            # next_sample = distr.sample()
            next_sample = model.projection (dec_last_hidden) 
            repeated_past_values = torch.cat(
                # (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
                 (repeated_past_values, next_sample), dim=1
            )
            future_samples.append(next_sample*repeated_scale + repeated_loc)
            # print(f'{k} --_> {next_sample.shape}')
        # print(f'{future_samples[0].shape}')
        concat_future_samples = torch.cat(future_samples, dim=1)
        # print(f'{concat_future_samples.shape}')
        return SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, model.config.prediction_length) + (model.config.output_size,),
            )
        )
