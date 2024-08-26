'''
Utility Treatment Routines for ML Predicting models
===================================================


This modules contains a set of routines for preparing data for insertion in transformers models used
for prediction of multivariate time series data. The classes are contaiend in the companion file `AIClasses.py` and are imported in this module.
The classes are contaiend in the companion file `AIClasses.py` and are imported in this module.

Utilities 
---------

'''
import os,sys
import math
import numpy as np
import numpy.linalg as lin  
import xarray as xr
import pandas as pd

import scipy.linalg as sc

# import matplotlib.pyplot as plt
# import datetime

import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# import transformers as tr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import zapata.computation as zcom
import zapata.data as zd
# import zapata.lib as zlib
import zapata.mapping as zmap
# import zapata.koopman as zkop
# import AIModels.AIClasses as zaic

def func_name():
    """
    :return: name of caller
    """
    return sys._getframe(1).f_code.co_name

def copy_dict(data, strip_values=False, remove_keys=[]):
    '''
    Copy dictionary

    Parameters
    ==========

    data: dict
        Dictionary to be copied
    strip_values: boolean
        If True, strip values
    remove_keys: list
        List of keys to be removed
    
    Returns
    =======

    out: dict
        Copied dictionary without the keys in `remove_keys`
    '''
    if type(data) is dict:
        out = {}
        for key, value in data.items():
            if key not in remove_keys:
                out[key] = copy_dict(value, strip_values=strip_values, remove_keys=remove_keys)
        return out
    else:
        return [] if strip_values else data
    
def select_field(INX,outfield,verbose=False):
    '''
    Select field for `outfield` in dict
    '''
    for i in INX.keys():
        if i == outfield:
            if verbose:
                print(INX[i])
            out = INX[i]['X']
            arealat = np.array(INX[i]['arealat'])
            arealon = np.array(INX[i]['arealon']) 
            centlon = np.array(INX[i]['centlon'])         
            break
    return out,arealat,arealon,centlon

def select_field_key(INX,outfield,dataname):
    '''
    Select field for `outfield` in dict according to `dataname`
    '''
    for i in INX.keys():
        if i == outfield:
            out = INX[i][dataname]      
            break
    return out

def select_field_eof(INX,outfield):
    '''
    Select eof-related fields for `outfield` in dict
    '''
    for i in INX.keys():
        if i == outfield:
            print(f'{func_name()} ---> Extracting EOF data for {INX[i]["field"]}')
            U = INX[i]['udat']
            V = INX[i]['vdat'] 
            S = INX[i]['sdat'] 
            break
    return U,S,V

def select_area(area,ZZ):
    '''
    Select area for dataset ZZ
    
    Parameters
    ==========

    area: string
        Area to be analyzed, possible values are
        (EUROPE only implemented)

        * 'TROPIC': Tropics
        * 'GLOBAL': Global
        * 'PACTROPIC': Pacific Tropics
        * 'WORLD': World
        * 'EUROPE': Europe
        * 'NORTH_AMERICA': North America
        * 'NH-ML': Northern Hemisphere Mid-Latitudes
    
    ZZ: xarray dataset

    Returns
    =======

    Z: xarray dataset

    '''
    shift_Centlon = 180
    # central pacific coordinates
    if area == 'TROPIC':  
        arealat=(35,-35)
        arealon=np.array([0, 360.])
    elif area == 'GLOBAL':
        arealat=(60,-60)
        arealon=np.array([120,290.] ) 
    elif area == 'PACTROPIC':
        arealat=(25,-25)
        arealon=np.array([180,290.]) 
    elif area == 'WORLD':
        arealat=(70,-30)
        arealon=[0,360.]
    elif area == 'NH-ML':
        arealat=(90,20)
        arealon=[0,360.]
    elif area == 'EUROPE':
        # Require central longitude to 0 lon
        shift_Centlon = 0
        arealat=(70,30)
        arealon=[-15, 50.]
    elif area == 'NORTH_AMERICA':
        arealat=(70,25)
        arealon=[200, 310.]
    else:
        print(f'No area defined')
        raise ValueError 
    
    arealat = np.array(arealat)
    arealon = np.array(arealon)

    if area =='EUROPE':
        print('Use Greenwich centered coordinates with centlat=0')
        ZZ1 = ZZ.sel(lat=slice(70,30),lon=slice(340,360))
        ZZ1 = ZZ1.assign_coords({'lon':ZZ1.lon -360})
        Z = xr.concat((ZZ1, ZZ.sel(lat=slice(70,30),lon=slice(0,50))), dim='lon')
        del ZZ1
    else:
        print('Use Pacific centered coordinates with centlat=180')
        Z = ZZ.sel(lat=slice(arealat[0],arealat[1]), lon=slice(arealon[0],arealon[1]))
        Z = Z.assign_coords(lon = Z.lon-180.)
        arealon = arealon - 180
    return Z

def make_matrix(S,SMOOTH,normalization,shift,**kwargs):
    '''
    Make matrix for SVD analysis
    
    Parameters
    ==========
    S: xarray dataset
        Dataset with the field to be analyzed
    SMOOTH: boolean
        If True, smooth the data
    normalization: string
        Type of normalization to be applied
    shift: string
        Type of shift to be applied to select the data
    dropnan: String
        Option to drop NaN values from Xmat
        True -- Uses indexed dropna in Xmat
        False -- Uses standard dropna in Xmat from Xarray
    detrend: boolean
        If True, detrend the data
    
    Returns
    =======

    X: xarray dataset
        Math Matrix with EOF coefficients
    
    '''
    # Default values
    default_values = {'dropnan': False, 'detrend' : False}
    mv = {**default_values, **kwargs}
    print(mv)
    #Weight cosine of latitude
    WS=np.cos(S.lat*math.pi/180)

    # Create Xmat Matrix
    if mv['dropnan']:
        X=zcom.Xmat(S*WS,dims=('lat','lon'),option='DropNaN')
        print('Option DropNaN -- Shape of Xmat',X.A.shape)
    else:
        X=zcom.Xmat(S*WS,dims=('lat','lon'))
        X.A = X.A.dropna(dim='z')
        print('Option DropNaN False -- Shape of Xmat',X.A.shape)
    
    
    # detrend data
    if mv['detrend']:
        print('make_matrix: -- Detrending data')
        X.detrend(axis=1)

    # Smooth data
    # Put label on the last month of the window
    if SMOOTH:
        X.A = X.A.rolling(time=3,center=False,min_periods=3).mean().dropna(dim='time')
        
    # Create anomalies
    X.anom(option=normalization)

    if shift == 'SCRAM':
        X.A = X.A[:,np.random.permutation(np.arange(len(X.A.time)))].assign_coords(time=S.time[:-2])

    return X

def make_eof(X, mr,eof_interval=None):
    '''
    Compute EOF analysis on data matrix
    
    Parameters
    ==========
    
    X: X matrix
        Data matrix in `X` format
    mr: int
        Number of modes to be retained
    eof_interval: list
        List with the starting and ending date for EOF analysis
    
    Returns
    =======
    
    mr: int
        Number of modes retained, in case the number of modes is less than rank
        Set to `math.inf` to keep all modes
    var_retained: float
        Percentage of variance retained
    udat: numpy array
        Matrix of EOF modes
    vdat: numpy array
        Matrix of EOF coefficients
    sdat: numpy array
        Vector of singular values
    
    '''
    xdat = X.A.data

    #Check eof interval
    if eof_interval is not None:        
        eofstart = X.A.time.to_index().get_loc(eof_interval[0])
        eofend = X.A.time.to_index().get_loc(eof_interval[1])
        print(f'make_eof: -- EOF interval defined -- Using data from {eof_interval[0]} to {eof_interval[1]}')
        print(f'make_eof: -- EOF interval defined -- Using data from {eofstart} to {eofend}')
        udatx, sdatx, _ = sc.svd(xdat[:,eofstart:eofend], full_matrices=False,lapack_driver='gesvd')
    else:
        udatx, sdatx, _ = sc.svd(xdat[:,:], full_matrices=False)
    # lmr = lin.matrix_rank(xdat)
    lmr = matrix_rank_light(xdat,sdatx)
    mr = min(mr,lmr)
    print(f'  Number of SVD modes retained {mr}, rank of matrix {lmr}')

    var = sdatx**2
    var_retained = sum(var[0:mr])/sum(var)
    print(f'Variance Retained {var_retained:.2f} out of possible {len(var)} modes')
    print(f' Condition number {max(sdatx)/min(sdatx)}')
    
    #keep only mr modes
    # S Field
    udat=udatx[:,0:mr]
    sdat=sdatx[0:mr]
    # the columns of vhdatx contain the coefficients of the field (standardized to unit variance)
    # vhdat=vhdatx[0:mr,:]
    # cofficients non standardized directly from projection on EOF
    vdat=udat.T @ xdat
 
    
    return mr, var_retained, udat,vdat, sdat
def make_field(*args,**kwargs):
    '''
    Make field for analysis
    
    Parameters
    ==========
    
    area: string (postional argument)
        Area to be analyzed, possible values are
        
        * 'TROPIC': Tropics
        * 'GLOBAL': Global
        * 'PACTROPIC': Pacific Tropics
        * 'WORLD': World
        * 'EUROPE': Europe
        * 'NORTH_AMERICA': North America
        * 'NH-ML': Northern Hemisphere Mid-Latitudes

    var: string (keyword argument)
        Variable to be analyzed
    level: string (keyword argument)
        Level to be analyzed
    period: string (keyword argument)
        Period to be analyzed
    version: string (keyword argument)
        Version of the dataset
    loc: string (keyword argument)
        Location of the dataset
        
    Returns
    =======
    
    Z: xarray dataset
        Field to be analyzed
    arealat: numpy array
        Latitude limits of the field
    arealon: numpy array
        Longitude limits of the field
    
    '''
    # Default values
    default_values = {'var':'SST','level':'SST','period':'ANN','version':'V5','loc':None}
    merged_values = {**default_values, **kwargs}
    
    verdata = merged_values['version']
    match verdata:
        case 'V5':
            return make_field_V5(*args,**merged_values)
        case 'HAD':
            return make_field_HAD(*args,**merged_values)
        case _:
            print(f'Version {verdata} not defined')
            return None

 

def make_field_V5(area,**kwargs):
    '''
    Make field for analysis
    
    Parameters
    ==========
    
    area: string
        Area to be analyzed, possible values are
        
        * 'TROPIC': Tropics
        * 'GLOBAL': Global
        * 'PACTROPIC': Pacific Tropics
        * 'WORLD': World
        * 'EUROPE': Europe
        * 'NORTH_AMERICA': North America
        * 'NH-ML': Northern Hemisphere Mid-Latitudes

    var: string
        Variable to be analyzed
    level: string
        Level to be analyzed
    period: string
        Period to be analyzed
    version: string
        Version of the dataset
    loc: string
        Location of the dataset
        
    Returns
    =======
    
    Z: xarray dataset
        Field to be analyzed
    arealat: numpy array
        Latitude limits of the field
    arealon: numpy array
        Longitude limits of the field
    
    '''
    
    period = kwargs['period']
    version = kwargs['version']
    var = kwargs['var']
    level = kwargs['level']
    loc = kwargs['loc']

    
    shift_Centlon = 180

    #area='WORLD'
    # central pacific coordinates
    if area == 'TROPIC':  
        arealat=(35,-35)
        arealon=np.array([0, 360.])
    elif area == 'GLOBAL':
        arealat=(60,-60)
        arealon=np.array([120,290.] ) 
    elif area == 'PACTROPIC':
        arealat=(25,-25)
        arealon=np.array([180,290.]) 
    elif area == 'WORLD':
        arealat=(70,-30)
        arealon=[0,360.]
    elif area == 'NH-ML':
        arealat=(90,20)
        arealon=[0,360.]
    elif area == 'EUROPE':
        # Require central longitude to 0 lon
        shift_Centlon = 0
        arealat=(70,30)
        arealon=[-15, 50.]
    elif area == 'NORTH_AMERICA':
        arealat=(70,25)
        arealon=[200, 310.]
    else:
        print(f'No area defined')
    arealat = np.array(arealat)
    arealon = np.array(arealon)

    dd=zd.in_data(var,level,period=period,epoch=version, loc = loc,averaging=False,verbose=True)
    ZZ = dd[var.lower()]

    if area =='EUROPE':
        print('Use Greenwich centered coordinates with centlat=0')
        ZZ1 = ZZ.sel(lat=slice(70,30),lon=slice(340,360))
        ZZ1 = ZZ1.assign_coords({'lon':ZZ1.lon -360})
        Z = xr.concat((ZZ1, ZZ.sel(lat=slice(70,30),lon=slice(0,50))), dim='lon')
        del ZZ1
    else:
        print('Use Pacific centered coordinates with centlat=180')
        Z = ZZ.sel(lat=slice(arealat[0],arealat[1]), lon=slice(arealon[0],arealon[1]))
        Z = Z.assign_coords(lon = Z.lon-180.)
        arealon = arealon - 180
        
    
    del dd,ZZ
    print(f'Selecting field {var} for level {level} and area {area}')
    return Z, arealat, arealon, shift_Centlon

def make_field_HAD(area,**kwargs):
    '''
    Make field for analysis for HADSST
    
    Parameters
    ==========
    
    area: string
        Area to be analyzed, possible values are
        
        * 'TROPIC': Tropics
        * 'GLOBAL': Global
        * 'PACTROPIC': Pacific Tropics
        * 'WORLD': World
        * 'EUROPE': Europe
        * 'NORTH_AMERICA': North America
        * 'NH-ML': Northern Hemisphere Mid-Latitudes

    var: string
        Variable to be analyzed
    level: string
        Level to be analyzed
    period: string
        Period to be analyzed
    version: string
        Version of the dataset
    loc: string
        Location of the dataset
        
    Returns
    =======
    
    Z: xarray dataset
        Field to be analyzed
    arealat: numpy array
        Latitude limits of the field
    arealon: numpy array
        Longitude limits of the field
    
    '''
    period = kwargs['period']
    version = kwargs['version']
    var = kwargs['var']
    level = kwargs['level']
    loc = kwargs['loc']

    shift_Centlon = 180

    #area='WORLD'
    # central pacific coordinates
    if area == 'TROPIC':  
        arealat=(35,-35)
        arealon=np.array([0, 360.])
    elif area == 'GLOBAL':
        arealat=(60,-60)
        arealon=np.array([120,290.] ) 
    elif area == 'PACTROPIC':
        arealat=(25,-25)
        arealon=np.array([180,290.]) 
    elif area == 'WORLD':
        arealat=(70,-30)
        arealon=[0,360.]
    elif area == 'NH-ML':
        arealat=(90,20)
        arealon=[0,360.]
    elif area == 'EUROPE':
        # Require central longitude to 0 lon
        shift_Centlon = 0
        arealat=(70,30)
        arealon=[-15, 50.]
    elif area == 'NORTH_AMERICA':
        arealat=(70,25)
        arealon=[200, 310.]
    else:
        print(f'No area defined')
    arealat = np.array(arealat)
    arealon = np.array(arealon)

    dstart = '1/1/1870'
    dtend = '12/30/2020'    
    data_time = pd.date_range(start=dstart, end=dtend, freq='1MS')
    
    # Hist1950
    datadir = loc + '/Dropbox (CMCC)/ERA5/DATA/HADSST/'
    files = 'HadISST_sst.nc'    

    print(f'{datadir}  \n SST file --> \t {files} \n')
    ds = xr.open_dataset(datadir + files,use_cftime=None).drop_dims('nv').rename({'latitude': 'lat', 'longitude': 'lon'})
    
    sst_Pac = zmap.adjust_data_centlon(ds.sst)
    sst_Pac = sst_Pac.assign_coords({'lon':sst_Pac.lon+180.})
    # 

    #Fix discontinuity at dateline
    sst_Pac.loc[dict(lon=180.5)] = (sst_Pac.sel(lon=180-1.5)+sst_Pac.sel(lon=182.5))/2
    sst_Pac.loc[dict(lon=180+1.5)] = (sst_Pac.sel(lon=180-1.5)+sst_Pac.sel(lon=182.5))/2

    ZZ = sst_Pac.sel(lat=slice(arealat[0],arealat[1]),lon=slice(arealon[0],arealon[1])).assign_coords({'time': data_time})
    ZZ.data[abs(ZZ.data) > 100] = np.nan
    

    if area =='EUROPE':
        print('Use Greenwich centered coordinates with centlat=0')
        ZZ1 = ZZ.sel(lat=slice(70,30),lon=slice(340,360))
        ZZ1 = ZZ1.assign_coords({'lon':ZZ1.lon -360})
        Z = xr.concat((ZZ1, ZZ.sel(lat=slice(70,30),lon=slice(0,50))), dim='lon')
        del ZZ1
    else:
        print('Use Pacific centered coordinates with centlat=180')
        Z = ZZ.sel(lat=slice(arealat[0],arealat[1]), lon=slice(arealon[0],arealon[1]))
        Z = Z.assign_coords(lon = Z.lon-180.)
        arealon = arealon - 180
        
        
    
    del ZZ
    print(f'Selecting field {var} for level {level} and area {area}')
    return Z, arealat, arealon, shift_Centlon

def make_data(INX,params):
    '''
    Prepare data for analysis and concatenate as needed. Modify input `INX` dictionary
    by adding values for `scaler` and `index` for each field. `scaler` is the scaler used, `index` is the
    index of the data in the concatenated matrix INX.

    The Convention for indeces is that they point to the real date.
    If python ranges need to be defined then it must take into account the extra 1
    in the end of the range.

    
    Parameters
    ==========
    
    INX: dict
        Dictionary with the fields to be analyzed
    params: dict
        Dictionary with the parameters for the analysis
    
    Returns
    =======
    
    datain: numpy array
        Matrix with the data to be analyzed
    INX: dict
        Input dictionary updated with the information from the data analysis
    
    '''
    
    indstart = 0
    for num,i in enumerate(INX.keys()):
        print(f'\nProcessing field {INX[i]["field"]} that is {INX[i]["datatype"]}')
        _,_, vdat = select_field_eof(INX,i)
        indstart = indstart 
        indend = indstart + INX[i]['mr']
        print('vdat',vdat.shape)


        #Normalize Training Data
        if params['scaling'] == 'MaxMin':
            sstr = MinMaxScaler(feature_range=(-1,1))
        else:
            sstr = StandardScaler()
        if num == 0:
            datatr = sstr.fit_transform(vdat.T[params['train_period_start']:params['train_period_end']+1,:])
        else:
            datatr = np.concatenate((datatr,sstr.fit_transform(vdat.T[params['train_period_start']:params['train_period_end']+1,:])),axis=1)
        INX[i]['scaler_tr'] = sstr

        #Normalize Validation Data
        if params['scaling'] == 'MaxMin':
            ssva = MinMaxScaler(feature_range=(-1,1))
        else:
            ssva = StandardScaler()
        if num == 0:
            datava = ssva.fit_transform(vdat.T[params['val_period_start']:params['val_period_end']+1,:])
        else:
            datava = np.concatenate((datava,ssva.fit_transform(vdat.T[params['val_period_start']:params['val_period_end']+1,:])),axis=1)
        INX[i]['scaler_va'] = ssva

        #Normalize Test Data
        # Use the scaling of the training data
        if num == 0:
            datate = sstr.transform(vdat.T[params['test_period_start']:params['test_period_end']+1,:])
        else:
            datate = np.concatenate((datate,sstr.transform(vdat.T[params['test_period_start']:params['test_period_end']+1,:])),axis=1)
        
        INX[i]['scaler_te'] = sstr

        INX[i]['index'] = np.arange(indstart, indend)
       
        print(f'Added field {i} to feature input data')
        print(f'Index for field {INX[i]["field"]} are {indstart} and {indend}\n')
        print(f'Using {params["scaling"]} scaling') 
        print(f'Using  {INX[i]["mr"]}  EOFs for {INX[i]["var_retained"]} variance retained')

        indstart  = indend
    
    # Transform to Tensor
    datatr = torch.tensor(datatr, device=params['device'], dtype=params['t_type'])
    datava = torch.tensor(datava, device=params['device'], dtype=params['t_type'])
    datate = torch.tensor(datate, device=params['device'], dtype=params['t_type'])

    print(f'Training data shape {datatr.shape}')
    print(f'Validation data shape {datava.shape}')
    print(f'Testing data shape {datate.shape}')    
    return datatr, datava, datate, INX

def make_features(INX):
    '''
    Prepare features for analysis and compute features boundaries
    
    Parameters
    ==========
    
    INX: dict
        Dictionary with the fields to be analyzed
    
    Returns
    =======
    
    num_features: int
        Number of features
    
    m_limit: list
        List with the boundaries of the features
    
    '''
    num_features = 0
    m_limit = []

    for num,ii in enumerate(INX.keys()):
        print(f'\nProcessing field {INX[ii]["field"]} that is {INX[ii]["datatype"]}')
        num_features += INX[ii]['mr']
        m_limit.append(INX[ii]['mr'])
        print(f'Limit for {INX[ii]["field"]} is {INX[ii]["mr"]}')
    print(f'Total number of features  {num_features}')    
    return num_features, m_limit

def make_data_base(InputVars, period='ANN', version='V5', SMOOTH=False, normalization='STD', \
                   eof_interval = None, detrend=False,\
                   shift='ERA5', case=None, datatype='Source_data', location='DDIR'):
    '''
    Organize data variables in data base `INX`

    Parameters
    ==========

    InputVars: list
        List of variables to be analyzed
    period: string
        Period to be analyzed
    version: string
        Version of the dataset
    SMOOTH: boolean
        If True, smooth the data
    normalization: string
        Type of normalization to be applied
    eof_interval: list
        List with the starting and ending date for EOF analysis
    detrend: boolean
        If True, detrend the data
    shift: string
        Type of shift to be applied to select the data
    case: string
        Case to be analyzed
    datatype: string
        Type of data to be analyzed, either `Source_data` or `Target_data`
    location: string
        Data directory
    
    Returns
    =======

    INX: dict
        Dictionary with the fields to be analyzed. The dictionary is organized as follows:
        INX = {'id1':{'case':case,'datatype': datatype,'field':invar.name,'level':inlevel,
        'centlon':centlon,'arealat':arealat, 'arealon':arealon, 
        X':X,'xdat':xdat,'mr':mr,'var_retained':varr,'udat':udat,'vdat':vdat,'sdat':sdat}}

    
    '''
    INX = {}
    for invar in InputVars:
        for inlevel in invar.levels:
            S,arealat,arealon, centlon = make_field(invar.area,var=invar.name,level=inlevel,period=period,version=version,loc=location)
            X = make_matrix(S,SMOOTH,normalization,shift,dropnan=False,detrend=detrend)
          
            if eof_interval is None:
                print(f'No EOF interval defined -- Using all data')
                mr, varr, udat, vdat,sdat = make_eof(X,invar.mr)
            else:
                print(f'EOF interval defined -- Using data from {eof_interval[0]} to {eof_interval[1]}')
                mr, varr, udat, vdat,sdat = make_eof(X,invar.mr,eof_interval=eof_interval)
            dv = {'case':case,'area':invar.area, 'datatype': datatype,'field':invar.name,'level':inlevel, 'centlon':centlon,\
                  'arealat':arealat, 'arealon':arealon, 'X':X,'mr':mr,'var_retained':varr,'udat':udat,'vdat':vdat,'sdat':sdat}
           
            id = (invar.name+inlevel).upper()
            INX.update( {id:dv})

            print(f'Added field `{invar.name}` with identification `{id}` to data base')
            del S
    return INX 

def init_weights(m):
    ''' Initialize weights with uniform ditribution'''
    
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    ''' Count Model Parameters'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    '''Compute Execution time pr epoch'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_time_features(data_time, start,device):
    '''
    Create the past features for monthly means

    Parameters
    ==========

    data_time: xarray dataset
        Time data
    start: datetime
        Starting date
    device: string
        Device to be used for computation
    
    Returns
    =======

    pasft: torch tensor
        Tensor with the past features
        Three levels:

        * The first level is the month sequence
        * The second level is the seasonal cycle
        * The third is the year
    
    
    '''
    
               
    pasft = torch.zeros(len(data_time),3,device=device,dtype=torch.float32)
    tt = data_time.year
    
    #Months
    pasft[:,0] = torch.tensor((data_time.month - 1) / 11.0 - 0.5)

    #Seasons
    pasft[:,1] = torch.tensor((data_time.quarter-2.5)/4)
    
    #Years    
    pasft[:,2] = torch.tensor((data_time.year-data_time.year[0])/(data_time.year[-1]-data_time.year[0]))  
    
    return pasft

def rescale (params, PDX, out_train0, out_val0, out_test0):
    ''''
    Rescale data to original values, according to scaling choice in

    Parameters
    ==========

    params: dict
        Dictionary with the parameters for the analysis
    PDX: dict
        Dictionary with the information for the fields to be analyzed
    out_train0: numpy array
        Training data
    out_val0: numpy array
        Validation data
    out_test0: numpy array
        Test data

    Returns
    =======

    out_train: numpy array
        Rescaled training data
    out_val: numpy array
        Rescaled validation data
    out_test: numpy array
        Rescaled test data
    true: numpy array
        Original data
    
    '''

    Tpredict = params['Tpredict']
    out_train = np.ones(out_train0.shape)
    out_val = np.ones(out_val0.shape)
    out_test = np.ones(out_test0.shape)

    
    for num,i in enumerate(PDX.keys()):
        print(f'\nProcessing field {PDX[i]["field"]} that is {PDX[i]["datatype"]}')
        print(f'Number of modes retained {PDX[i]["mr"]}')
        print(PDX[i]['vdat'].T.shape)
        indloc = PDX[i]['index']
        
        if num == 0:
            for t in range(Tpredict):
                out_train[:,t,indloc] = PDX[i]['scaler_tr'].inverse_transform(out_train0[:,t,indloc])
                out_val[:,t,indloc] = PDX[i]['scaler_va'].inverse_transform(out_val0[:,t,indloc])
                out_test[:,t,indloc] = PDX[i]['scaler_te'].inverse_transform(out_test0[:,t,indloc])
            true = PDX[i]['vdat'].T
        else:
            print(out_train.shape)
            for t in range(Tpredict):
                out_train[:,t,indloc] = PDX[i]['scaler_tr'].inverse_transform(out_train0[:,t,indloc])
                out_val[:,t,indloc] = PDX[i]['scaler_va'].inverse_transform(out_val0[:,t,indloc])
                out_test[:,t,indloc] = PDX[i]['scaler_te'].inverse_transform(out_test0[:,t,indloc])
            true = np.concatenate((true,PDX[i]['vdat'].T),axis=1)
    return out_train,out_val,out_test,true 
    
def make_dyn_verification(ver_f, area, dyn_cases, dynddr, times, dyn_startdate, dyn_enddate, filever):
    '''
    Make dynamic verification data
    
    Parameters
    ==========
    
    ver_f: numpy array
        Verification data
    area: string
        Area to be analyzed
    dyn_cases: list
        List of cases to be analyzed
    dynaddr: string
        Address of the dynamic data
    times: numpy array
        Time data
    dyn_startdate: string
        Starting date for the data
    dyn_enddate: string
        Ending date for the data
    filever: string
        Name of the file to be written
    
    Returns
    =======
    
    ver_d: numpy array
        Verification data in numpy format
    
    '''
    endindex = np.where(times == dyn_enddate)
    startindex = np.where(times == dyn_startdate)

    ndyn = int(endindex[0]-startindex[0]+1)
    # ngrid = INX['T2MT2M']['X'].A.shape[0]
    # dynddr =  homedir + '/Dropbox (CMCC)/ERA5/SEASONAL_'+ ver_f
    print(f'Starting date for verification of dynamic data {dyn_startdate} and ending date {dyn_enddate}')
    print(f'Number of months for verification {ndyn}')
    print(f'Verification field {ver_f}')

    lead_length = 6

    match ver_f:
        case 'SST':
            vername_f = 'ssta'
        case 'T2M':
            vername_f = 't2a'
        case _ :
            print(f'Verification field {ver_f} not defined')
            raise ValueError
        
    try:
        out = xr.open_dataset(filever).stack(z=('lat','lon'))[vername_f]
        print(f'Verification data loaded from file {filever}')
    except:
        print(f'Verification data not found -- Creating file {filever}')
        # ver_d = np.ones((ndyn,6,ngrid))
        for cen in dyn_cases:
            for lead in range(lead_length):
                vfiles = [f"{dynddr}/{cen[ver_f]}_{cen['center']}_{k}_{lead+1}.nc" for k in cen['system']]
               
                for index, value in enumerate(vfiles):
                    print(lead,index, value)
                    ddd = xr.open_dataset(value).rename({'longitude':'lon','latitude':'lat'}).sel(time=slice(times[startindex[0]+lead][0],times[endindex[0]-6+lead][0])).mean(dim='number')#.persist()
                    print(f'Variable selected ---> {vername_f}, size {ddd[vername_f].shape}')
                    out_field = select_area(area,ddd).stack(z=('lat','lon')).transpose('time','z')[vername_f]
                    if index == 0:
                        result = out_field
                    else:
                        result = xr.concat([result,out_field],dim='time')
                del ddd,out_field
                print(f'\nAt lead {lead}, Result shape {result.shape}, times {result.time[0].data} to {result.time[-1].data}')
                if lead == 0:
                    out_time = result.time
                    out = xr.full_like(result, 1).expand_dims(dim={"lead":range(1,lead_length+1)}, axis=1).assign_coords(time=out_time).copy()
                    out[:,lead,:] = result.assign_coords(time=out_time)
                    print(f'\n Lead 0 --> {out.shape},{result.shape}')
                else:
                    out[:,lead,:] = result.assign_coords(time=out_time)
                    

        print(f'Finished analysis for lead {lead_length} {out.shape}')
        out.unstack().to_netcdf(filever,engine='h5netcdf') 
    
    return out
      
def eof_to_grid_new(choice,field, verification, forecasts, times, INX=None, params=None, truncation=None):
    '''
    Transform from the eof representation to the grid representation,
    putting data in the format (Np, Tpredict,gridpoints) in stacked format .
    Where Np is the total number of cases that is given by 
    $N_p = L - TIN - Tpredict +1$
    L is the total length of the test period and  Tpredict is the number of lead times
    and gridpoints is the number of grid points in the field.
    All fields start at month 1 of the prediction.

    Parameters
    ==========

    choice: string
        Choice of data to be analyzed, either `test`, `validation` or `training`
    field: string
        Field to be analyzed
    verification: numpy array
        Verification data from reanalysis
    forecasts: numpy array
        Forecasts data from the network
    times: DateTime
        Time data for entire period
    INX: dict
        Dictionary with the information for the fields to be analyzed
    truncation: int
        Number of modes to be retained in the observations
    
    Returns
    =======

    The routines returns arrays in the format of xarray datasets
    as (Np,lead,grid points) in stacked format

    Fcst: xarray dataset
        Forecast data in grid representation
    Obs: xarray dataset
        Observations data in grid representation
    Ver: xarray dataset
        Verification data in grid representation
    Per: xarray dataset
        Persistence data in grid representation

    '''
    if (INX or params) is None:
        print(f'INX/dtix/params not defined')
        return None
    # Get the pattern EOF for the requested field
    zudat = INX[field]['udat']
    # Get the `X` matrix for the field as a template -- it contains the time information
    XZ = INX[field]['X'].A
    # Length of the prediction period
    Tpredict = params['Tpredict']
    # Length of input sequence
    TIN = params['TIN']
    # Length of prediction training
    T = params['T']

    # Transform the observation to gridpoint
    # if truncation is not None truncate to `truncation`
    if truncation is not None:
        vereof = truncation
        # zudat = zudat[:,:vereof]
        Obs = XZ.copy(data=zudat[:,:vereof]@(verification[:,:vereof].T))
        print(f'Verification using truncated EOFs with {vereof} modes')
        Obs_for_Per = XZ.copy(data=zudat[:,:vereof]@(verification[:,:vereof].T))
        print(f'Persistence using truncated EOFs with {vereof} modes')
    else:
        Obs = XZ#.copy(data=zudat@(verification.T))
        print(f'Verification using entire field {Obs.shape})')
        Obs_for_Per = XZ.copy()
        print(f'Persistence using entire field {Obs_for_Per.shape}')
    print(f'Obs from {Obs.time[0].data} to {Obs.time[-1].data}\n')

    
    #compute template arrays
    match choice:
        case 'test':
            starttest = params['test_period_start']  
            endtest = params['test_period_end'] 
            first_forecast = params['test_first_fcs']
            last_forecast =  params['test_last_fcs']
        case 'validation':
            starttest = params['val_period_start']   
            endtest = params['val_period_end'] 
            first_forecast = params['val_first_fcs']
            last_forecast =  params['val_last_fcs']
        case 'training':
            starttest = 0
            endtest =  params['train_period_end']
            first_forecast = params['train_first_fcs']
            last_forecast =  params['train_last_fcs'] 
        case _:
            print(f'Choice {choice} not defined in {eof_to_grid.__name__}')
            return None
    
    # Define number of cases (forecasts) to be analyzed 
    # Works only for TIN=1,T=1
    Np =last_forecast - first_forecast 
    print(f' Number of cases {Np}')

    tclock = Obs.time.data
    
    # Align the verification data with the first foreacast day date
    # The last forecast is already taken into account in `test_period_end`
    # starttest=starttest-1
    
    print(f'Observation selected for {choice} \n ')
    print(f'Obs from {Obs.time[first_forecast].data} to {Obs.time[last_forecast].data}\n')
    print(f'Verification for first forecast at {first_forecast} to {last_forecast}')
    print(f'Verification time for `{choice}` from first month forecast {tclock[first_forecast]} to {tclock[last_forecast]}')  

    flead = np.arange(0,Tpredict+1)
    
    # Assemble arrays for verification on grid
    # The Observation IC start one time step before the forecast
    # because the forecast contains only from lead 1 to lead Tpredict
    # The IC for the forecast (lead=0) is empty
    Xoo = Obs.isel(time=slice(first_forecast-1,last_forecast+Tpredict)).drop_vars('month').transpose('time','z')
    Xoo_Per = Obs_for_Per.isel(time=slice(first_forecast-1,last_forecast+Tpredict)).drop_vars('month').transpose('time','z')
    Tmp = Obs.isel(time=slice(first_forecast-1,last_forecast-1)).expand_dims({"lead": Tpredict+1}).drop_vars('month').assign_coords({"lead": flead}).transpose('time','lead','z')
    kz = np.ones((Np,Tpredict+1,len(Xoo.z))  )


    for i in range(Np):
        for j in range(Tpredict+1):     
            kz[i,j,:] = Xoo.data[i+j,:]
            # print(f'Verification {i},{j} --> {i+j} {Xoo.time.data[i+j]}')
    print(f'Tmp shape {Tmp.shape}, {Tmp.time.data[0]} to {Tmp.time.data[-1]}')
    Ver = Tmp.copy(data=kz,deep=True)  
    print(f'Verification shape {Ver.shape}, {Ver.time.data[0]} to {Ver.time.data[-1]}')

    # Compute forecast and consider only the `Np` forecast that fit with the Tpredict period
    hhh = np.einsum('ijk,lk',forecasts,INX[field]['udat'])
    
    # Keep only the `Np` cases that are within the Tpredict period
    kzz = np.ones((Np,Tpredict+1,len(Tmp.z))  )
    kzz[:,0,:] = Xoo_Per[:Np,:].data # kz[:,0,:]
    
    for i in range(1,Tpredict+1):
        kzz[:,i,:] = hhh[:Np,i-1,:]
    Fcst = Tmp.copy(data=kzz,deep=True)
    print(f'Forecast shape {Fcst.shape}, {Fcst.time.data[0]} to {Fcst.time.data[-1]}')

    # Compute Persistence
    # Start from the previous time step with respect to the first forecast
    kzz = np.ones((Np,Tpredict+1,len(Xoo.z))  )
    # Xoo = Obs_for_Per.isel(time=slice(first_forecast-1,last_forecast+Tpredict)).drop_vars('month').transpose('time','z')
    for i in range(Np):
        for j in range(Tpredict+1):     
            kzz[i,j,:] = Xoo_Per.data[i,:]
    Per = Tmp.copy(data=kzz,deep=True)
    print(f'Persistence shape {Per.shape}, {Per.time.data[0]} to {Per.time.data[-1]}')
    

    return Fcst, Ver, Per, Obs.transpose()


def matrix_rank_light(X,S):
    '''
    Compute the rank of a matrix using the singular values
    
    Parameters
    ==========

    X: numpy array
        Matrix to be analyzed
    
    S: numpy array
        Singular values of the matrix
    
    Returns
    =======
    
    rank: int
        Rank of the matrix
    
    '''

    rtol = max(X.shape[-2:]) * np.finfo(S.dtype).eps
      
    tol = S.max(axis=-1, keepdims=True) * rtol
   

    return np.count_nonzero(S > tol, axis=-1)
        
