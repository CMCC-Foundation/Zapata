'''
Utility Treatment Routines for ML Predicting models
===================================================


This modules contains a set of routines for preparing data for insertion in transformers models used
for prediction of multivariate time series data. The classes are contaiend in the companion file `AIClasses.py` and are imported in this module.
The classes are contaiend in the companion file `AIClasses.py` and are imported in this module.

Utilities 
---------

'''
import os,sys,re,gc
import math
import numpy as np
# import numpy.linalg as lin  
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
import AIModels.AIClasses as zaic




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
def get_arealat_arealon(area):
    '''
    Obtain lat lon extent for various choices of areas

    Parameters
    ==========

    area: string
        Area to be analyzed, possible values are
        

        * 'TROPIC': Tropics (-35,35), [0,360]
        * 'GLOBAL': Global (-60,60), [120,290]
        * 'PACTROPIC': Pacific Tropics (-25,25), [180,290]
        * 'WORLD': World (-70,70), [0,360]
        * 'EUROPE': Europe (30,70), [-15,50]
        * 'NORTH_AMERICA': North America (25,70), [200,310]
        * 'NH-ML': Northern Hemisphere Mid-Latitudes (20,90), [0,360]

    '''
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
        arealat=(70,-70)
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

    return arealat, arealon
    
def select_field(INX,outfield,verbose=False):
    '''
    Select field for `outfield` in dict

    PARAMETERS
    ==========

    INX: dict
        Dictionary with the fields to be analyzed
    outfield: string
        Field to be analyzed
    verbose: boolean
        If True, print the field information
    
    RETURNS
    =======

    out: numpy array
        Field data
    arealat: numpy array
        Latitude limits of the field
    arealon: numpy array
        Longitude limits of the field
    centlon: numpy array
        Central longitude of the field
    
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

    PARAMETERS
    ==========

    INX: dict
        Dictionary with the fields to be analyzed
    outfield: string
        Field to be analyzed
    dataname: string
        Key to be analyzed
    
    RETURNS
    =======

    out: numpy array
        Field data
    
    '''
    for i in INX.keys():
        if i == outfield:
            out = INX[i][dataname]      
            break
    return out

def select_field_eof(INX,outfield):
    '''
    Select eof-related fields for `outfield` in dict

    PARAMETERS
    ==========

    INX: dict
        Dictionary with the fields to be analyzed
    outfield: string
        Field to be analyzed
    
    RETURNS
    =======

    U: numpy array
        EOF modes
    S: numpy array
        Singular values
    V: numpy array
        EOF coefficients
    
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
    arealat, arealon = get_arealat_arealon(area)

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
        udatx, sdatx, vdatx = sc.svd(xdat[:,eofstart:eofend], full_matrices=False,lapack_driver='gesvd')
    else:
        udatx, sdatx, vdatx = sc.svd(xdat[:,:], full_matrices=False)
    
    # lmr = lin.matrix_rank(xdat)
    lmr = matrix_rank_light(xdat,sdatx)
    var = sdatx**2

    if isinstance(mr, float) and 0 < mr < 1.0:
        print(f'Target Variance {mr}')
        total_variance = var.sum()
        csum = np.cumsum(var)
        new_mr = np.searchsorted(csum, mr * total_variance) + 1
        mr = int(min(new_mr, lmr))
        var_retained = csum[mr - 1] / total_variance
        print(f'  Number of SVD modes retained {mr}, rank of matrix {lmr}')
        print(f'  Variance Retained {var_retained:.2f} out of possible {len(var)} modes')
    else:
        
        mr = min(mr,lmr)
        print(f'  Number of SVD modes retained {mr}, rank of matrix {lmr}')       
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
    print(f'Use cofficients non standardized directly from projection on EOF')
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

    arealat, arealon = get_arealat_arealon(area)

    dd=zd.in_data(var,level,period=period,epoch=version, loc = loc,averaging=False,verbose=True)
    ZZ = dd[var.lower()]

    if area =='EUROPE':
        print('Use Greenwich centered coordinates with centlat=0')
        ZZ1 = ZZ.sel(lat=slice(70,30),lon=slice(340,360))
        ZZ1 = ZZ1.assign_coords({'lon':ZZ1.lon -360})
        Z = xr.concat((ZZ1, ZZ.sel(lat=slice(70,30),lon=slice(0,50))), dim='lon')
        shift_Centlon = 0
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
    arealat, arealon = get_arealat_arealon(area)

    dstart = '1/1/1870'
    dtend = '12/30/2020'    
    data_time = pd.date_range(start=dstart, end=dtend, freq='1MS')
    
    # Hist1950
    datadir = loc + '/DATA/HADSST/'
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

def normalize_training_data(params, vdat, period='train',scaler=None,feature_scale=1):
    '''
    Normalize training data

    PARAMETERS
    ==========

    params: dict
        Dictionary with the parameters for the analysis
    vdat: numpy array
        Data to be normalized
    period: string
        Period to be analyzed
    scaler: object
        Scaler object
    feature_scale: float
        Feature scale

    RETURNS
    =======

    datatr: numpy array
        Normalized data
    sstr: object
        Scaler object
    
    '''
    ps = period + '_period_start'
    pe = period + '_period_end'
    
    if scaler is None:
        if params['scaling'] == 'MaxMin':
            sstr = MinMaxScaler(feature_range=(-1 ,1))
            datatr = sstr.fit_transform(vdat.T[params[ps]:params[pe] + 1, :])
        elif params['scaling'] == 'Standard':
            sstr = StandardScaler()
            datatr = sstr.fit_transform(vdat.T[params[ps]:params[pe] + 1, :])
        elif params['scaling'] == 'Identity':
            sstr = zaic.IdentityScaler()
            datatr = sstr.fit_transform(vdat.T[params[ps]:params[pe] + 1, :])
        elif params['scaling'] == 'SymScaler':
            sstr = zaic.SymmetricFeatureScaler(feature_scales=feature_scale)
            datatr = sstr.fit_transform(vdat.T[params[ps]:params[pe] + 1, :])
        else:
            raise ValueError(f'Wrong scaling defined for {params["scaling"]}')
    else:
        datatr = scaler.transform(vdat.T[params[ps]:params[pe]+1,:])
        sstr = scaler
    return datatr, sstr

def make_data(INX,params):
    '''
    Prepare data for analysis and concatenate as needed. Modify input `INX` dictionary
    by adding values for `scaler` and `index` for each field. `scaler` is the scaler used, `index` is the
    index of the data in the concatenated matrix INX.

    The Convention for indeces is that they point to the real date.
    If python ranges need to be defined then it must take into account the extra 1
    in the end of the range.

    
    PARAMETERS
    ==========
    
    INX: dict
        Dictionary with the fields to be analyzed
    params: dict
        Dictionary with the parameters for the analysis
    
    RETURNS
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
        #Choose scaling
        if params['scaling'] == 'SymScaler':
            print(f'Using Symmetric Scaler')
            # Use the feature scale in the Symmetric Scaler
            var12 = INX[i]['var_retained']
            ss = INX[i]['sdat']
            vartot = sum(ss**2)/var12
            feat_scale = ss**2/vartot
        else:
            print(f'Using {params["scaling"]} scaling')
            feat_scale = 1

           

        #Normalize Training Data
        tmp, sstr = normalize_training_data(params, vdat, period='train',feature_scale=feat_scale)
        if num == 0:
            datatr = tmp
        else:
            datatr = np.concatenate((datatr,tmp),axis=1)
        
        INX[i]['scaler_tr'] = sstr

        #Normalize Validation Data
        tmp, ssva = normalize_training_data(params, vdat, period='val', feature_scale=feat_scale)
        if num == 0:
            datava = tmp
        else:
            datava = np.concatenate((datava,tmp),axis=1)
       
        INX[i]['scaler_va'] = ssva

        #Normalize Test Data
        # Use the scaling of the training data
        tmp, _ = normalize_training_data(params, vdat, period='test',scaler=sstr, feature_scale=feat_scale)
        if num == 0:
            datate = tmp
        else:
            datate = np.concatenate((datate,tmp),axis=1)
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
    
    PARAMETERS
    ==========
    
    INX: dict
        Dictionary with the fields to be analyzed
    
    RETURNS
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
                   eof_interval = None, detrend=False, \
                   shift='ERA5', case=None, datatype='Source_data', location='DDIR'):
    '''
    Organize data variables in data base `INX`

    PARAMETERS
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
    
    RETURNS
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
                print(f'No EOF interval defined -- Using all data, using {invar.mr} modes')
                mr, varr, udat, vdat,sdat = make_eof(X,invar.mr)
            else:
                print(f'EOF interval defined -- Using data from {eof_interval[0]} to {eof_interval[1]}')
                mr, varr, udat, vdat,sdat = make_eof(X,invar.mr,eof_interval=eof_interval)
            
            dv = {'case':case,'area':invar.area, 'datatype': datatype,'field':invar.name,'level':inlevel, 'centlon':centlon,\
                  'arealat':arealat, 'arealon':arealon, 'X':X,'mr':mr,'var_retained':varr,'udat':udat,'vdat':vdat,'sdat':sdat}
            
            if invar.dropX:
                print(f'Dropping X matrix')
                del dv['X']
                gc.collect()

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
    '''Compute Execution time per epoch'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_time_features(data_time, start,device):
    '''
    Create the past features for monthly means

    PARAMETERS
    ==========

    data_time: xarray dataset
        Time data
    start: datetime
        Starting date
    device: string
        Device to be used for computation
    
    RETURNS
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
    # pasft[:,2] = torch.tensor((data_time.year-data_time.year[0])/(data_time.year[-1]-data_time.year[0]))  
    lent = data_time.shape[0]
    pasft[:,2] = torch.tensor(np.arange(lent)/lent)
    return pasft

def rescale (params, PDX, out_train0, out_val0, out_test0,verbose=True):
    ''''
    Rescale data to original values, according to scaling choice in

    PARAMETERS
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

    RETURNS
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
        if verbose:
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
            if verbose:
                print(out_train.shape)
            for t in range(Tpredict):          
                out_train[:,t,indloc] = PDX[i]['scaler_tr'].inverse_transform(out_train0[:,t,indloc])
                out_val[:,t,indloc] = PDX[i]['scaler_va'].inverse_transform(out_val0[:,t,indloc])
                out_test[:,t,indloc] = PDX[i]['scaler_te'].inverse_transform(out_test0[:,t,indloc])
            true = np.concatenate((true,PDX[i]['vdat'].T),axis=1)
    return out_train,out_val,out_test,true 
    
# def make_dyn_verification(ver_f, area, dyn_cases, dynddr, times, dyn_startdate, dyn_enddate, filever):
#     '''
#     Make dynamic verification data
    
#     PARAMETERS
#     ==========
    
#     ver_f: numpy array
#         Verification data
#     area: string
#         Area to be analyzed
#     dyn_cases: list
#         List of cases to be analyzed
#     dynaddr: string
#         Address of the dynamic data
#     times: numpy array
#         Time data
#     dyn_startdate: string
#         Starting date for the data
#     dyn_enddate: string
#         Ending date for the data
#     filever: string
#         Name of the file to be written
    
#     RETURNS
#     =======
    
#     ver_d: numpy array
#         Verification data in numpy format
    
#     '''
#     endindex = np.where(times == dyn_enddate)
#     startindex = np.where(times == dyn_startdate)

#     ndyn = int(endindex[0]-startindex[0]+1)
#     # ngrid = INX['T2MT2M']['X'].A.shape[0]
#     # dynddr =  homedir + '/Dropbox (CMCC)/ERA5/SEASONAL_'+ ver_f
#     print(f'Starting date for verification of dynamic data {dyn_startdate} and ending date {dyn_enddate}')
#     print(f'Number of months for verification {ndyn}')
#     print(f'Verification field {ver_f}')

#     lead_length = 6

#     match ver_f:
#         case 'SST':
#             vername_f = 'ssta'
#         case 'T2M':
#             vername_f = 't2a'
#         case _ :
#             raise ValueError(f'Verification field {ver_f} not defined')
        
#     try:
#         out = xr.open_dataset(filever).stack(z=('lat','lon'))[vername_f]
#         print(f'Verification data loaded from file {filever}')
#         print(f'Verification data shape {out.shape}, times {out.time[0].data} to {out.time[-1].data}')
#     except:
#         print(f'Verification data not found -- Creating file {filever}')
#         # ver_d = np.ones((ndyn,6,ngrid))
#         for cen in dyn_cases:
#             for lead in range(lead_length):
#                 vfiles = [f"{dynddr}/{cen[ver_f]}_{cen['center']}_{k}_{lead+1}.nc" for k in cen['system']]
               
#                 for index, value in enumerate(vfiles):
#                     print(lead,index, value)
#                     ddd = xr.open_dataset(value).rename({'longitude':'lon','latitude':'lat'}).sel(time=slice(times[startindex[0]+lead][0],times[endindex[0]-6+lead][0])).mean(dim='number')#.persist()
#                     print(f'Variable selected ---> {vername_f}, size {ddd[vername_f].shape}')
#                     out_field = select_area(area,ddd).stack(z=('lat','lon')).transpose('time','z')[vername_f]
#                     if index == 0:
#                         result = out_field
#                     else:
#                         result = xr.concat([result,out_field],dim='time')
#                 del ddd,out_field
#                 print(f'\nAt lead {lead}, Result shape {result.shape}, times {result.time[0].data} to {result.time[-1].data}')
#                 if lead == 0:
#                     out_time = result.time
#                     out = xr.full_like(result, 1).expand_dims(dim={"lead":range(1,lead_length+1)}, axis=1).assign_coords(time=out_time).copy()
#                     out[:,lead,:] = result.assign_coords(time=out_time)
#                     print(f'\n Lead 0 --> {out.shape},{result.shape}')
#                 else:
#                     out[:,lead,:] = result.assign_coords(time=out_time)
                    

#         print(f'Finished analysis for lead {lead_length} {out.shape}')
#         print(f'Verification data shape {out.shape}, times {out.time[0].data} to {out.time[-1].data}')
#         out.unstack().to_netcdf(filever,engine='h5netcdf') 
    
#     return out
      

def matrix_rank_light(X,S):
    '''
    Compute the rank of a matrix using the singular values
    
    PARAMETERS
    ==========

    X: numpy array
        Matrix to be analyzed
    
    S: numpy array
        Singular values of the matrix
    
    RETURNS
    =======
    
    rank: int
        Rank of the matrix
    
    '''

    rtol = max(X.shape[-2:]) * np.finfo(S.dtype).eps
      
    tol = S.max(axis=-1, keepdims=True) * rtol
   

    return np.count_nonzero(S > tol, axis=-1)
        
def CPRSS(observations, climate, forecasts):
    '''
    Compute the CRPSS score for ensemble forecasts
    with respect to the climate reference.

    Positive CRPSS (0 < CRPSS ≤ 1): Indicates that your forecast model has better skill than the reference model.
        Closer to 1: Greater improvement over the reference.
    
    Zero CRPSS (CRPSS = 0): Your forecast model performs equally to the reference model.
        Negative CRPSS (CRPSS < 0): Indicates that your forecast model performs worse than the reference model.
        More Negative: Greater underperformance compared to the reference.
    
    PARAMETERS
    ==========

    observations: numpy array
        Observations data
    
    forecasts: numpy array
        Forecasts data
    
    climate: numpy array
        Reference data
    
    RETURNS
    =======

    score: float
        CRPS score
    
    '''
    from properscoring import crps_ensemble
    scorefor = crps_ensemble(observations, forecasts)
    scorref = crps_ensemble(observations, climate)
    # print(f"CRPS: {score[0]:.2f} °C")
    return 1 - scorefor/scorref

import re

import re

import re

def transform_strings(strings):
    """
    Transforms a list of strings by extracting the repeating pattern 
    (2, 3, or 7 characters long) from each string.
    
    Parameters:
        strings (list of str): List of input strings.
        
    Returns:
        list of str: Transformed strings with only the repeating pattern.
    """
    result = []
    # The regex:
    #  ^                      : start of string
    #  (?P<grp>(?:.{2}|.{3}|.{7})) : named group 'grp' that matches exactly 2, 3, or 7 characters
    #  (?:\1)*                : zero or more repetitions of the exact same substring captured in 'grp'
    #  $                      : end of string
    pattern = re.compile(r'^(?P<grp>(?:.{2}|.{3}|.{7}))(?:\1)*$')
    
    for string in strings:
        match = pattern.fullmatch(string)
        if match:
            result.append(match.group('grp'))
        else:
            result.append(string)  # Append the original string if no pattern found

    return result

def make_fcst_array(startdate, enddate, leads, data):
    '''
    Make forecast array for verification.
    The input uses the xarray DataArray format
    of dimension (ntim, lead, z) where z is a stacked coordinate.

    The output is an xarray DataArray with the time, lead, and z dimensions,
    and the valid_time coordinate as a 2D array.

    The lead time starts from 0 as the last element of
    the input sequence to leads-1.
    
    PARAMETERS
    ==========
    
    startdate: string
        Starting date for the forecast
    enddate: string
        Ending date for the forecast
    leads: int
        Number of leads, including the IC
    data: xarray DataArray
        DataArray with the forecast data
    
    RETURNS
    =======
    
    out: xarray DataArray
        Forecast array for verification
    
    '''
    if len(data.shape) == 4:
        ntim, member, nlead, n = data.shape
        print(f'Number of times {ntim}, number of members {member}, number of leads {nlead}, number of features {n}')
    else:
        ntim, nlead, n = data.shape
        print(f'Number of times {ntim}, number of leads {nlead}, number of features {n}')

    time = pd.date_range(startdate, enddate, freq="MS")  # Start-of-month freq
    print(time)
    
    # Validation checks
    if ntim != len(time):
        raise ValueError(f"Number of times {ntim} does not match length of time {len(time)}")
    if nlead != leads:
        raise ValueError(f"Number of leads {nlead} does not match expected leads {leads}")
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Data must be an xarray DataArray, but got {type(data)}")

    # Create the valid_time coordinate
    valid_time = xr.DataArray(
        np.array([
            [t + pd.DateOffset(months=int(l)) for l in np.arange(nlead)]
            for t in time
        ]),
        dims=["time", "lead"],
        coords={"time": time, "lead": np.arange(nlead)},
        name="valid_time"
    )

    # Assign the valid_time coordinate to the data
    data = data.assign_coords(valid_time=valid_time)

    return data

def eof_to_grid(field, forecasts, startdate, enddate, params=None, INX=None, truncation=None):
    '''
    Transform from the eof representation to the grid representation,
    putting data in the format (Np, Tpredict,gridpoints) in stacked format .
    Where Np is the total number of cases that is given by 
    $N_p = L - TIN - Tpredict +1$
    L is the total length of the test period and  Tpredict is the number of lead times
    and gridpoints is the number of grid points in the field.
    All fields start at month 1 of the prediction.


    PARAMETERS
    ==========

    field: string
        Field to be analyzed
    forecasts: numpy array
        Forecasts data from the network
    startdate: int
        Starting date for the forecast (IC)
    enddate: int
        Ending date for the forecast
    params: dict
        Dictionary with the parameters for the analysis
    INX: dict
        Dictionary with the information for the fields to be analyzed
    truncation: int
        Number of modes to be retained in the observations
    
    RETURNS
    =======

    The routines returns arrays in the format of xarray datasets
    as (Np,lead,grid points) in stacked format

    Fcst: xarray dataset
        Forecast data in grid representation with dims (Np,lead,grid points)
    Per: xarray dataset
        Persistence data in grid representation with dims (Np,lead,grid points)

    '''
    if INX is None or params is None:
        raise ValueError("INX or params not defined")
    # Get the pattern EOF for the requested field
    zudat = INX[field]['udat']
    # Get the `X` matrix for the field as a template -- it contains the time information
    XZ = INX[field]['X'].A
    # Length of the prediction period
    Tpredict = params['Tpredict']
    # Length of input sequence
    TIN = params['TIN']
    # Length of prediction 
    T = params['T']

    # Transform the observation to gridpoint
    # if truncation is not None truncate to `truncation`
    if truncation is not None:
        xpro = XZ.data.T@zudat[:,:truncation]
        Obs = XZ.copy(data=zudat[:,:truncation]@xpro.T)
        print(f'Verification using truncated EOFs with {INX[field]["mr"]} modes')
    else:
        Obs = XZ
        print(f'Verification using entire field {Obs.shape})')
    print(f'Obs from {Obs.time[0].data} to {Obs.time[-1].data}\n')

    
    # Define number of cases (forecasts) to be analyzed
    # Calculate the number of months
    Np = (enddate.year - startdate.year) * 12 + (enddate.month - startdate.month) + 1
    print(f'Number of cases {Np}')

    tclock = Obs.time.data
    
    # Align the verification data with the first foreacast date     
    print(f'Obs from {startdate} to {enddate}\n')
    print(f'Forecasts for IC at {startdate} to {enddate} and prediction time of {Tpredict} months')
    
    flead = np.arange(0,Tpredict+1)
    
    # Assemble arrays for verification on grid
    # The Observation IC start one time step before the forecast
    # because the forecast contains only from lead 1 to lead Tpredict
    # The IC for the forecast (lead=0) is empty
    Tmp = Obs.sel(time=slice(startdate.strftime('%Y-%m-%d'),enddate.strftime('%Y-%m-%d'))).expand_dims({"lead": Tpredict+1}).drop_vars('month').assign_coords({"lead": flead}).transpose('time','lead','z')

    # Compute forecast and consider only the `Np` forecast that fit with the Tpredict period
    hhh = np.einsum('ijk,lk',forecasts,INX[field]['udat'])
    
    # Keep only the `Np` cases that are within the Tpredict period
    kzz = np.ones((Np,Tpredict+1,len(Tmp.z))  )
    kzz[:,0,:] = Tmp[:Np,0,:].data # kz[:,0,:]
    
    for i in range(1,Tpredict+1):
        kzz[:,i,:] = hhh[:Np,i-1,:]
    Fcst = Tmp.copy(data=kzz,deep=True)
    print(f'Forecast shape {Fcst.shape}, {Fcst.time.data[0]} to {Fcst.time.data[-1]}')

    # Compute Persistence
    # Start from the previous time step with respect to the first forecast
    kzz = np.ones((Np,Tpredict+1,len(Tmp.z))  )
   
   
    for i in range(Np):
        for j in range(Tpredict+1):     
            kzz[i,j,:] = Tmp.data[i,0,:]
    Per = Tmp.copy(data=kzz,deep=True)
    print(f'Persistence shape {Per.shape}, {Per.time.data[0]} to {Per.time.data[-1]}')
    

    return Fcst, Per, Obs.transpose('time','z')
    # routine to advance a Timestamp of one month
def advance_months(ts, n=1):
    '''
    Advance or reduce a Timestamp by n months
    
    PARAMETERS
    ==========
    
    ts: pd.Timestamp or str
        Timestamp to be modified. If str, it should be in the YYYY-MM-DD format.
    n: int
        Number of months to advance (positive) or reduce (negative)
    
    RETURNS
    =======
    
    pd.Timestamp or str
        Modified timestamp. If input was a string, output will be a string in the YYYY-MM-DD format.
    '''
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
        return (ts + pd.DateOffset(months=n)).strftime('%Y-%m-%d')
    return ts + pd.DateOffset(months=n)

def project_dyn(data, INX, field, truncation=None):
    '''
    Project the dynamic data into the EOF space
    
    PARAMETERS
    ==========
    
    data: xarray dataset
        Dynamic data
    INX: dict
        Dictionary with the information for the fields to be analyzed
    field: string
        Field to be analyzed
    truncation: int
        Number of modes to be retained in the observations
    
    RETURNS
    =======
    
    out: xarray dataset
        Projected data, in stacked format with NaN values
    
    '''
    
    # Get the pattern EOF for the requested field
    zudat = INX[field]['udat']
    # Get the `X` matrix for the field as a template -- it contains the time information
    XZ = INX[field]['X'].A
    # Transform the observation to gridpoint
    # if truncation is not None truncate to `truncation`
    if truncation is not None:
        xzudat = XZ.isel(time=np.arange(truncation)).copy(data=zudat[:,:truncation]).unstack().stack(z=('lat','lon'))
        xNEW = data.unstack().stack(z=('lat','lon'))
        projNEW = np.einsum('ijk,lk',np.where(np.isnan(xNEW), 0, xNEW),np.where(np.isnan(xzudat), 0, xzudat))
        tmpdata = np.einsum('ijk,kl',projNEW,xzudat)
        recon_DATA = xNEW.copy(data=tmpdata)
        print(f'Verification using truncated EOFs with {truncation} modes')
    else:
        recon_DATA = data.unstack().stack(z=('lat','lon'))
        print(f'Verification using entire field {data.shape})')
    
    
    return recon_DATA
    
def make_dyn_verification_new(ver_f, area, dyn_cases, dynddr, times, filever):
    '''
    Make dynamic verification data. Read all time levels for the GCM data
    
    PARAMETERS
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
    filever: string
        Name of the file to be written
    
    
    RETURNS
    =======
    
    ver_d: numpy array
        Verification data in numpy format
    
    '''
    

    lead_length = 6

    match ver_f:
        case 'SST':
            vername_f = 'ssta'
        case 'T2M':
            vername_f = 't2a'
        case 'Z500':
            vername_f = 'za'
        case _ :
            raise ValueError(f'Verification field {ver_f} not defined')
        
    try:
        out = xr.open_dataset(filever).stack(z=('lat','lon'))[vername_f]
        print(f'Verification data loaded from file {filever}')
        print(f'Verification data shape {out.shape}, times {out.time[0].data} to {out.time[-1].data}')
    except:
        print(f'Verification data not found -- Creating file {filever}')
        # 
        for cen in dyn_cases:
            for lead in range(lead_length):
                vfiles = [f"{dynddr}/{cen[ver_f]}_{cen['center']}_{k}_{lead+1}.nc" for k in cen['system']]
                
                for index, value in enumerate(vfiles):
                    print(lead,index, value)
                    match ver_f:
                        case 'SST':
                            ddd = xr.open_dataset(value).rename({'longitude':'lon','latitude':'lat'}).mean(dim='number')
                        case 'T2M':
                            ddd = xr.open_dataset(value).rename({'longitude':'lon','latitude':'lat'}).mean(dim='number')
                        case 'Z500':
                            ddd = xr.open_dataset(value).rename({'longitude':'lon','latitude':'lat','forecast_reference_time':'time'}).mean(dim='number')
                            # Define new latitude and longitude coordinates
                            new_lat = np.arange(-90, 90.22, 0.25)  # Adjust based on dataset bounds
                            new_lon = np.arange(0, 360, 0.25)
                            # Perform interpolation
                            # ddd = ddd.interp(lat=new_lat, lon=new_lon, method="linear").isel(forecastMonth=0,pressure_level=0).drop_vars({'forecastMonth','pressure_level'})
                            ddd = ddd.sel(lat=slice(None, None, -1)).interp(lat=new_lat, lon=new_lon, method="cubic",kwargs={"fill_value": None}).sel(lat=slice(None, None, -1)).drop_vars({'forecastMonth','pressure_level'}).squeeze()
                        case _ :
                            raise ValueError(f'Verification field {ver_f} not defined')
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
                    print(f'Lead 0 --> {out.shape},{result.shape}')
                else:
                    out[:,lead,:] = result.assign_coords(time=out_time)
                    

        print(f'Finished analysis for lead {lead_length} {out.shape}')
        print(f'Verification data shape {out.shape}, times {out.time[0].data} to {out.time[-1].data}')
        
        out.unstack().to_netcdf(filever,engine='h5netcdf') 
    
    return out
def compute_increments(tensor, axis=0):
    '''
    Take the difference of the torch tensor along the specified axis and output the initial value
    
    PARAMETERS
    ==========
    
    tensor: torch tensor
        Tensor to be analyzed
    axis: int
        Axis along which to compute the differences
    
    RETURNS
    =======
    
    diff: torch tensor
        Tensor with the differences along the specified axis
    init_value: torch tensor
        Initial value along the specified axis
    
    '''
    print(f'Computing differences for input tensor along axis {axis}')
    
    # Initialize the difference tensor with the same shape as the input tensor
    diff = torch.zeros_like(tensor)
    
    # Compute the differences along the specified axis
    diff = torch.diff(tensor, dim=axis, prepend=tensor.select(axis, 0).unsqueeze(axis))
    
    # Extract the initial value along the specified axis
    init_value = tensor.select(axis, 0)
    
    return diff, init_value
def cumsum_with_init(differences, init_value):
    """
    Computes the cumulative sum for a PyTorch tensor of differences with specific initial values.
    The differences tensor has dimensions (time, nlead, neof), and the initial values tensor
    has dimensions (neof). The initial values are added to the first time level of each slice (lead)
    and then accumulated using the cumulative sum.

    Parameters:
    - differences: torch.Tensor
        Input tensor containing the differences with dimensions (time, nlead, neof).
    - init_value: torch.Tensor
        A 1D tensor representing the initial values, with a size matching the last dimension (neof).

    Returns:
    - result: torch.Tensor
        The resulting tensor after computing the cumulative sum with initial values.
    """
    # Ensure init_value matches the last dimension of differences
    if init_value.size(0) != differences.size(-1):
        raise ValueError("init_value must match the size of the last dimension in the input tensor.")

    # Create a copy of the differences tensor to avoid modifying the input
    result = differences.clone()

    # Add init_value to the first time level of each lead slice
    result[0] += init_value

    # Compute the cumulative sum along the time dimension (axis=0)
    result = torch.cumsum(result, dim=0)

    return result


def select_fcst(IC, my_data):
    '''
    Select the forecast data for the given initial condition IC from the xarray my_data dataset.

    PARAMETERS
    ==========

    IC: string
        Initial condition for the forecast
    my_data: xarray dataset
        Forecast data
    
    RETURNS
    =======

    ds_single_init: xarray dataset
        Forecast data for verification with coordinate "time" as the valid_time
    
    '''
    init_time_sel = IC
    ds_single_init = my_data.sel(time=init_time_sel, drop=False)

    # Turn valid_time into a 1D coordinate named "time"
    ds_single_init = ds_single_init.assign_coords(time=ds_single_init.valid_time)
    ds_single_init = ds_single_init.swap_dims({"lead": "time"})
    ds_single_init = ds_single_init.drop_vars("valid_time")

    # Debugging: Check coordinate presence
    # print("Coordinates in my_data:", list(my_data.coords))
    # print("Coordinates in ds_single_init before assignment:", list(ds_single_init.coords))

    # Check if 'z' is a MultiIndex in my_data
    z_index = my_data.indexes["z"]  # This is the actual pd.MultiIndex
    if isinstance(z_index, pd.MultiIndex):
        # Drop existing lat, lon if they exist
        ds_single_init = ds_single_init.drop_vars(["lat", "lon"], errors="ignore")

        # Extract lat and lon from the MultiIndex, then assign as coordinates
        lat_vals = z_index.get_level_values("lat")
        lon_vals = z_index.get_level_values("lon")

        ds_single_init = ds_single_init.assign_coords(
            lat=("z", lat_vals),
            lon=("z", lon_vals)
        )

        # Rebuild the MultiIndex for 'z' with original level names
        ds_single_init = ds_single_init.set_index(z=z_index.names)

    return ds_single_init.drop_vars("lead", errors="ignore")
def variance_features(INX):
    '''
    retur the variance of the features
    
    PARAMETERS
    ==========
    
    INX: dict
        Dictionary with the information for the fields to be analyzed
    
    RETURNS
    =======
    
    ssvar: numpy array
        Variance of the features
    
    '''
    ssvar = []
    for i, feature in enumerate(INX.keys()):
        tmp = INX[feature]['sdat']
        print(f'Processing field {feature} with shape {tmp.shape}')
        ss = tmp**2 / sum(tmp**2)
        ssvar.append(ss)
    ssvar = np.concatenate(ssvar, axis=0)

    return ssvar
import os

def create_subdirectory(parent_dir, subdirectory_name):
    """
    Create a subdirectory within the specified parent directory if it does not exist.

    Parameters
    ----------
    parent_dir : str
        The path to the parent directory.
    subdirectory_name : str
        The name of the subdirectory to create.

    Returns
    -------
    str
        The full path of the created or existing subdirectory.
    """
    subdirectory_path = os.path.join(parent_dir, subdirectory_name)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
        print(f"Directory created: {subdirectory_path}")
    else:
        print(f"Directory already exists: {subdirectory_path}")

    return subdirectory_path
def _set_directory(file_path):
    '''
    Set the directory for `file_path` and create it if it does not exist.

    Parameters
    ==========

    file_path: string
        Relative path to the data directory
    
    Returns
    =======
    
    homedir: string
        Root directory for the data
    drop_home: string
        Relative path to the data directory
    
    '''
    
    homedir = os.path.expanduser("~")
    target = homedir+file_path
    try:
        os.makedirs(target)
        print('Creating Directory ',homedir)
    except FileExistsError:
        print(f'Directory {target} already exists')
    return target

def eof_to_grid_new(field, forecasts, startdate, enddate, params=None, INX=None, truncation=None):
    '''
    Transform from the EOF representation to the grid representation.
    The routine now supports forecasts arrays with an extra ensemble dimension.
    
    For the original case, forecasts has shape (Np, T, n_eofs) and the
    output forecast dataset has dims (Np, lead, gridpoints). Now, if forecasts
    has shape (Np, K, T, n_eofs), the output forecast dataset will have dims
    (Np, member, lead, gridpoints).
    
    PARAMETERS
    ==========
    
    field: string
        Field to be analyzed
    forecasts: numpy array
        Forecast data from the network. Expected shape is either (Np, T, n_eofs)
        or (Np, K, T, n_eofs), where K is the ensemble size.
    startdate: datetime-like
        Starting date for the forecast (initial condition)
    enddate: datetime-like
        Ending date for the forecast
    params: dict
        Dictionary with the parameters for the analysis. Must include 'Tpredict', 'TIN', and 'T'
    INX: dict
        Dictionary with information for the fields to be analyzed (including the EOF patterns)
    truncation: int, optional
        Number of modes to be retained in the observations
    
    RETURNS
    ========
    
    Fcst: xarray DataArray
        Forecast data in grid representation.
        * If forecasts is 3D, dims are (time, lead, z).
        * If forecasts is 4D, dims are (time, member, lead, z).
    Per: xarray DataArray
        Persistence data in grid representation, with the same dims as Fcst.
    Obs: xarray DataArray
        Observation data in grid representation, with dims (time, z)
    '''
    if INX is None or params is None:
        raise ValueError("INX or params not defined")
    
    # Get the EOF patterns and template X matrix
    zudat = INX[field]['udat']
    XZ = INX[field]['X'].A
    Tpredict = params['Tpredict']
    TIN = params['TIN']
    T = params['T']

    # Transform the observation to gridpoint
    if truncation is not None:
        xpro = XZ.data.T @ zudat[:, :truncation]
        Obs = XZ.copy(data=zudat[:, :truncation] @ xpro.T)
        print(f'Verification using truncated EOFs with {INX[field]["mr"]} modes')
    else:
        Obs = XZ
        print(f'Verification using entire field {Obs.shape}')
    print(f'Obs from {Obs.time[0].data} to {Obs.time[-1].data}\n')
    
    # Determine number of forecast cases
    Np = (enddate.year - startdate.year) * 12 + (enddate.month - startdate.month) + 1
    print(f'Number of cases {Np}')
    
    print(f'Obs from {startdate} to {enddate}\n')
    print(f'Forecasts for IC at {startdate} to {enddate} and prediction time of {Tpredict} months')
    
    flead = np.arange(0, Tpredict+1)
    
    # Build the observation template (no ensemble dimension)
    Tmp = Obs.sel(time=slice(startdate.strftime('%Y-%m-%d'),
                             enddate.strftime('%Y-%m-%d'))
                 ).expand_dims({"lead": Tpredict+1}).drop_vars('month'
                 ).assign_coords({"lead": flead}).transpose('time', 'lead', 'z')
    
    # Check the forecasts dimensions and process accordingly
    if forecasts.ndim == 4:
        # forecasts shape: (Np, K, T, n_eofs)
        ensemble_size = forecasts.shape[1]
        # Reconstruct grid forecasts for each ensemble member:
        hhh = np.einsum('nktj,lj->nktl', forecasts, INX[field]['udat'])
        print('Allocate array for forecasts with shape (Np, member, Tpredict+1, gridpoints)')
        kzz = np.ones((Np, ensemble_size, Tpredict+1, len(Tmp.z)))
        # For lead 0, use the observation for all ensemble members
        obs_lead0 = Tmp[:Np, 0, :].data  # shape (Np, gridpoints)
        kzz[:, :, 0, :] = np.repeat(obs_lead0[:, np.newaxis, :], ensemble_size, axis=1)
        # For leads 1 to Tpredict, fill with reconstructed forecasts
        for lead in range(1, Tpredict+1):
            kzz[:, :, lead, :] = hhh[:Np, :, lead-1, :]
        # Create an xarray DataArray with dims (time, member, lead, z)
        Fcst = xr.DataArray(
            kzz,
            dims=["time", "member", "lead", "z"],
            coords={"time": Tmp.time.data[:Np],
                    "member": np.arange(ensemble_size),
                    "lead": flead,
                    "z": Tmp.z.data}
        )
    elif forecasts.ndim == 3:
        # Original behavior for forecasts shape: (Np, T, n_eofs)
        hhh = np.einsum('ijk,lk->ijl', forecasts, INX[field]['udat'])
        kzz = np.ones((Np, Tpredict+1, len(Tmp.z)))
        kzz[:, 0, :] = Tmp[:Np, 0, :].data
        for lead in range(1, Tpredict+1):
            kzz[:, lead, :] = hhh[:Np, lead-1, :]
        Fcst = xr.DataArray(
            kzz,
            dims=["time", "lead", "z"],
            coords={"time": Tmp.time.data[:Np],
                    "lead": flead,
                    "z": Tmp.z.data}
        )
    else:
        raise ValueError("Forecasts array must be 3D or 4D.")

    print(f'Forecast shape {Fcst.shape}, {Fcst.time.data[0]} to {Fcst.time.data[-1]}')
    
    # Compute persistence by repeating the initial condition (lead=0) across all leads
    if Fcst.ndim == 4:
        persistence_data = np.repeat(Fcst.data[:, :, 0:1, :], Tpredict+1, axis=2)
        Per = xr.DataArray(
            persistence_data,
            dims=Fcst.dims,
            coords=Fcst.coords
        )
    else:
        persistence_data = np.repeat(Fcst.data[:, 0:1, :], Tpredict+1, axis=1)
        Per = xr.DataArray(
            persistence_data,
            dims=Fcst.dims,
            coords=Fcst.coords
        )
    
    return Fcst, Per, Obs.transpose('time', 'z')
# Assuming Ft.time and Dt.time are arrays of datetime64, create an arrays that has only the dates that are
# in both arrays and get also the indeces of the original arrays
def get_common_dates(Ft, Dt):
    """
    Get common dates between two xarray datasets and their indices.
    
    Parameters
    ----------
    Ft : xarray.DataArray
        First dataset with a time dimension.
    Dt : xarray.DataArray
        Second dataset with a time dimension.
    
    Returns
    -------
    common_dates : numpy.ndarray
        Array of common dates.
    Ft_indices : numpy.ndarray
        Indices of common dates in the first dataset.
    Dt_indices : numpy.ndarray
        Indices of common dates in the second dataset.
    """
    common_dates = np.intersect1d(Ft.time.values, Dt.time.values)
    Ft_indices = np.where(np.isin(Ft.time.values, common_dates))[0]
    Dt_indices = np.where(np.isin(Dt.time.values, common_dates))[0]
    
    return common_dates, Ft_indices, Dt_indices