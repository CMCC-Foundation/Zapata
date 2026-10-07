"""

*Zapata* A revolutionary library for analysis and plotting of meteorological data.  

The mapping is based on cartopy and GEOCAT libraries from NCAR.
It uses `xarray` as a basic data structure. 

The module `data` contains the information on the data banks. The routine `DataGrid` in `data` must be modified to the location of the basic data for each installation.

*Zapata* contains computation modules and plotting modules. Examples of working Jupyterlab Notebooks are in `docs.examples_notebook`
    
**computation** :   
    Routines for averaging and various computations
    
**data** :  
    Information on data sets and routines to get data from the data sets

**mapping** :   
    Mapping routines based on *Cartopy*
    
**lib** :   
    Utilities for the rest of the modules.

**colormap** :
    Routines to use colormap in xml format

**koopman** :
    Routines for Koopman decomposition and eigenvalues handling

    Version 2.1
"""
import warnings
warnings.simplefilter('ignore')