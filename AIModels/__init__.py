"""

*AIModels* is a Python package for the designing ML networks of atmospheric and ocean data. It is based on the *xarray* package. 

It uses `xarray` as a basic data structure. 


*AIModels* contains several modules with classes and utility routines
The plotting modules are based on `matplotlib` and `cartopy`. 
The computation modules are based on `numpy` and `scipy`.
 
**ClimFormer** :   
    Class for ClimFormer network
    
**ClimLSTM** :  
    Class for ClimLSTM network

**LocalInformer** :   
    Class for LocalInformer network, a modified version of Informer for time series from HuggingFace
    
**ModelTraining** :   
    Class for training, validate and inference the models

**UtilPlot** :
    Routines for plotting and visualization of the data

**AIutil** :
    Utilities for the rest of the modules

    Version 2.1
"""
import warnings
warnings.simplefilter('ignore')