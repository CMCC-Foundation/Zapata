'''
Machine Learning
================

Routines and methods for machine learning

This module contains tools and methods for machine learning applications in eather and climate
Classes 
-------
| **EarlyStopping**      Define early stopping for models


Examples
--------

>>> options = {}
'''

class EarlyStopping():
    '''
    This class creates the erly stopping criterion and wrote the model parameters.
    The `patience` parameter control how long it waits before stopping the run.

    Parameters
    ----------
    patience: array 
        Data array with features and samples
    delta: float
        DElata
    path: str
        Namae and Path to write model parameters
                Default: 'checkpoint.pt'
    trace_func:
        trace print function.
                Default: print
    verbose: Bool
        Amount of printing
                Default: False
    '''
    
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt',trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
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