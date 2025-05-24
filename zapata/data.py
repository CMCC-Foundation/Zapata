'''
A module to store and treat data
================================
in griglia nativa 1986-2020 giornaliera
/data/products/GLOBAL_REANALYSES/C-GLORSv7/DAILY_MONTHLY

in griglia nativa 1986-2020 mensile
/data/products/GLOBAL_REANALYSES/C-GLORSv7/MONTHLY
'''

import os
import numpy as np
import xarray as xr
import pandas as pd
import zapata.lib as lib
import netCDF4 as net
import docrep
# Uses DOCREP for avoiding copying docstring, contrary to the docs delete works
# only on one param at the time.

d = docrep.DocstringProcessor()


# NCEP Reanalysis standard pressure levels
# 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
# 1000  925  850  700  600, 500  400  300, 250, 200, 150, 100      50          10
#



# def read_month(dataset, vardir,var1,level,yy,mm,type,option,verbose=False):
#     """
#     A routine to read one month of data from various datasets.
    
#     This routine will read data one month at a time from various data sets
#     described in *DataGrid()*
    
#     Parameters
#     ----------
#     dataset :   
#         Name of the dataset, ``ERA5``, ``GPCP``  

#     vardir :   
#         Path to the dataset 

#     var1 :   
#         Variable to extract 

#     level :   
#         Level of the Variable   

#     yy :    
#         Year
    
#     mm :    
#         Month

#     type :   
#         Type of data to reay. Currently hardwired to ``npy``

#     option :    
#         'Celsius'     For temperature Transform to Celsius
    
#     verbose: 
#         Tons of Output
    
#     Returns
#     --------
    
#     average :
#         Monthly data. 
    
#     Examples
#     --------
    
#     >>> read_month('ERA5','.','Z','500',1979,12,'npy',[],verbose=verbose)
#     >>> read_month('GPCP','.','TPREP','SURF',1979,12,'nc',[],verbose=verbose) 
#     >>> read_month('ERA5','.','T','850',1979,12,'npy',option=Celsius,verbose=verbose)
#     """
#     info=DataGrid()
#     if dataset == 'ERA5':
#  #       def adddir(name,dir):
#  #   return dir +'/' + name.split('.')[0]+'.npy'
#         fil1=lib.adddir(lib.makemm(var1,str(level),yy,mm),info[dataset]['place'])
#         if verbose: print(fil1)
#         if var1 == 'T' and option == 'Celsius':
#             data1=np.load(fil1) - 273.16
#         else:
#             data1=np.load(fil1)
#     elif dataset == 'GPCP':       
#         file = info[dataset]['place'] + '/gpcp_cdr_v23rB1_y' + str(yy) + '_m' + '{:02d}'.format(mm) + '.nc'      
#         data1 = net.Dataset(file).variables["precip"][:,:]
#     else:
#         Print(' Error in read_month, datset set as {}'.format(dataset))
#     return data1

def date_param():
    """ 
    Data Bank to resolve Month and Season averaging information

    Examples
    --------
    >>> index = data_param()
    >>> mon=index['DJF']['month_index']

    """
    months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    DJF ={'label':'DJF','month_index':[12,1,2]}
    JFM ={'label':'JFM','month_index':[1,2,3]}
    AMJ ={'label':'AMJ','month_index':[4,5,6]}
    JJA ={'label':'JAS','month_index':[6,7,8]}
    JAS ={'label':'JAS','month_index':[7,8,9]}
    SON ={'label':'SON','month_index':[10,11,12]}
    ANN ={'label':'ANN','month_index':[i for i in range(1,13)]}
    JAN ={'label':'JAN','month_index':[1]}
    FEB ={'label':'FEB','month_index':[2]}
    MAR ={'label':'JAN','month_index':[3]}
    APR ={'label':'APR','month_index':[4]}
    MAY ={'label':'MAY','month_index':[5]}
    JUN ={'label':'JUN','month_index':[6]}
    JUL ={'label':'JUL','month_index':[7]}
    AUG ={'label':'AUG','month_index':[8]}
    SEP ={'label':'SEP','month_index':[9]}
    OCT ={'label':'OCT','month_index':[10]}
    NOV ={'label':'NOV','month_index':[11]}
    DEC ={'label':'DEC','month_index':[12]}
    out={'DJF':DJF,
        'JFM': JFM,
        'AMJ': AMJ,
        'JAS': JAS,
        'JJA': JJA,
        'SON': SON,
        'ANN': ANN,
        'JAN': JAN,
        'FEB': FEB,
        'MAR': MAR,
        'APR': APR,
        'MAY': MAY,
        'JUN': JUN,
        'JUL': JUL,
        'AUG': AUG,
        'SEP': SEP,
        'OCT': OCT,
        'NOV': NOV,
        'DEC': DEC,
        'MONTHS': months
        }
    return out

def DataGrid(option=None):
    """
    Routine that returns a Dictionary with information on the requested Data Set.

    Currently these data sets are supported   

    * ERA5 -- Subset of monthly data of ERA5   
    * GPCP -- Monthly data of precipitation data set 
    * OCRD'-- cGLORS renalaysis V7 daily 1986-2020
    * OCRM'-- cGLORS renalaysis V7 monthly 1986-2020

    Info can be retrieved as ``grid[dataset][var]['start']`` for the starting years.
    See source for full explanation of the content.

    Parameters
    ----------
    Option :       
        * 'Verbose'      Tons of Output   
        * 'Info'         Info on data sets  

    Examples
    --------

    >>> DataGrid('info')
    >>> dat = DataGrid('verbose')
    """

    homedir = os.path.expanduser("~")
    if option == 'verbose': print('Root Directory for Local Data ',homedir)
    
    U ={      'level': [10, 50, 100,150, 200,250,300,400,500,600,700,850,925,1000],
              'start': 1979,
              'end': 2018,
              'label': 'U',
              'longname': 'Zonal Wind',
              'factor':1
               }
    V ={      'level': [10,50,100,150,200,250,300,400,500,600, 700,850,925,1000],
              'start': 1979,
              'end': 2018,
              'label': 'V',
              'longname': 'Meridional Wind',
              'factor':1
               }
    T ={      'level': [10,50,100,150, 200,250,300,400, 500,600, 700,850,925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 4,
              'label': 'T',
              'longname': 'Temperature',
              'factor':1
               }
    W ={      'level': [10,50,100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'W',
              'longname': 'Vertical Velocity',
              'factor':1
               }
    Z ={      'level': [ 200, 500],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'Z',
              'longname': 'Geopotential Height',
              'factor':1
               }
    tp ={     'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 0.01,
              'label': 'TP',
              'longname': 'Precipitation',
              'factor':60
               }
    MSL ={     'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 10,
              'label': 'MSL',
              'longname': 'Mean Sea Level Pressure',
              'factor':1/100.
               }
    SST ={    'level': ['SURF'],
              'start': 1979,
              'end': 2018,
              'cv': 10,
              'label': 'SST',
              'longname': 'Sea Surface Tenperature',
              'factor':1
               }
    THETA ={      'level': [10,50,100,150, 200,250,300,400, 500,600, 700,850,925, 1000],
              'start': 1979,
              'end': 2018,
              'cv': 4,
              'label': 'T',
              'longname': 'Potential Temperature',
              'factor':1
               }
    dataera5={'nlat': 721,
              'nlon': 1440,
              'latvec':[i for i in np.arange(-90,90.1,0.25)],
              'lonvec': [i for i in np.arange(0,360.1,0.25)],
              'latnp': np.asarray([i for i in np.arange(-90,90.1,0.25)][::-1]),  # For plotting
              'lonnp': np.asarray([i for i in np.arange(0,360.1,0.25)]),
              'clim': homedir + '/Dropbox (CMCC)/ERA5/CLIM',
              'place': homedir +'/Dropbox (CMCC)/ERA5/DATA/ERA5_MM',
              'host': 'local',
              'source_url': 'http://confluence.ecmwf.int/display/CKB/ERA5+data+documentation#ERA5datadocumentation-Parameterlistings',
              'desc':'ERA5 Monthly Mean for U,V,T,W,SLP 1979-2018',
              'special_value': 9999.,
              'U': U,
              'T': T,
              'V': V,
              'W': W,
              'Z': Z,
              'tp': tp,
              'MSL': MSL,
              'SST': SST,
              'THETA': THETA
              }
    
    precip_gpcp ={  'level': 'SURF',
              'start': 1979,
              'end': 2018,
              'label': 'precip',
              'factor':1,
              'longname': 'Precipitation GPCP',
               }
    
    datagpcp={'nlat': 72,
              'nlon': 144,
              'latvec':[i for i in np.arange(-88.75,90.1,2.5)],
              'lonvec': [i for i in np.arange(1.25,360.1,2.5)],
              'latnp': np.asarray([i for i in np.arange(-88.75,90.1,2.5)][::-1]),  # For plotting
              'lonnp': np.asarray([i for i in np.arange(1.25,360.,2.5)]),
              'precip': precip_gpcp,
              'place': homedir +'/Dropbox (CMCC)/ERA5/DATA/GPCP/TPREP',
              'clim': homedir + '/Dropbox (CMCC)/ERA5/DATA/GPCP/TPREP',
              'desc': 'Precipitation from the GPCP Project',
              'source_url': 'http://gpcp.umd.edu/',
              'host': 'local'
              }

    ocean_monthly={
              'start':  1986,
              'end':    2020,
              'place': '/data/products/GLOBAL_REANALYSES/C-GLORSv7/MONTHLY',
              'names': 'CMCC-CM2-HR4-pi_1m_<year><month>01_<year><month>31_grid_T.nc',
              'desc': 'Ocean Reanalysis V7 Monthly',
              'source_url': '',
              'host': 'DSS'
              }
    ocean_daily={
              'start':  1986,
              'end':    2020,
              'place': '/data/products/GLOBAL_REANALYSES/C-GLORSv7/DAILY_MONTHLY',
              'names': 'CMCC-CM2-HR4-pi_1m_<year><month>01_<year><month>31_grid_T.nc',
              'desc': 'Ocean Reanalysis V7 Daily',
              'source_url': '',
              'host': 'DSS'
              }
    grid={'ERA5': dataera5,
          'GPCP': datagpcp,
          'OCRD': ocean_daily,
          'OCRM': ocean_monthly
         }
    if option == 'info':
        for i in list(grid.keys()):
            print(grid[i]['desc'])
            print(grid[i]['place'])
            print(grid[i]['host'])
            print(grid[i]['source_url']+'\n')
        return
    
    return grid


# def readvar_grid(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False):
#     """
#     Read Variable from data sets
    
#     Parameters
#     ----------

#     region :    
#         *globe* for global maps, or [east, west, north, south]
#         for limited region, longitude 0-360
#     dataset :   
#          name of data set
#     var :   
#          variable name
#     level : 
#         level, either a value or 'SURF' for surface fields

#     season :    
#         Month ('JAN') or season (,'DJF') or annual 'ANN')
     
#     Celsius :   
#         True/False for temperature transform to Celsius
    
#     verbose :   
#         True/False -- tons of output

#     Returns
#     -------

#     xdat : numpy    
#         array data 
#     nlon :  
#         Number of longitudes
#     nlat :  
#         Number of Latitudes
#     lat :   
#         Latitudes
#     lon :   
#         Longitudes

#     Examples
#     --------

#     >>> readvar_grid(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False)
#     >>> readvar_grid(region='globe',dataset='ERA5',var='SST',level='SURF',season='JAN',Celsius=True,verbose=False)
#     """
    
#     vardir = '.'

#     dat=date_param()

#     grid = DataGrid()
#     nlat = grid[dataset]['nlat']
#     nlon = grid[dataset]['nlon']

#     lat = grid[dataset]['latnp']
#     lon = grid[dataset]['lonnp']
#     #Correct for longitude in ERA5
#     if dataset =='ERA5':
#         lon=lon[:-1]
#     sv=None
#     try:
#         sv=grid[dataset]['special_value']
#         if verbose: print('  Using Special Value ---->', sv)
#     except:
#         print('  Special Value not defined for dataset {}'.format(dataset))
        
#     ys=grid[dataset][var]['start']
#     ye=grid[dataset][var]['end']
#     ys=1979
#     ye=2018
#     lname=grid[dataset][var]['longname']

#     nyears= ye-ys
#     years =[i for i in range(ys,ye+1)]

#     factor=grid[dataset][var]['factor']

#     xdat= np.zeros([nlat,nlon,nyears+1])

#     for tim in years:
#         itim =years.index(tim)
#         dat=date_param()
#         mon=dat[season]['month_index']
#         if verbose:
#             print(' Plotting ' + var + ' from dataset ' + dataset)
#             print('Printing year  ', tim)
#         #
#         if len(mon) > 1:
#             if verbose: print('Mean on these months: {}'.format(mon))
#             temp= np.zeros([nlat,nlon,len(mon)])
#             for k in range(len(mon)):
#                 temp[:,:,k]=read_month(dataset,vardir,var,level,tim,mon[k],'npy',[],verbose=verbose) 
#             xdat[:,:,itim]=np.mean(temp,axis=2)
#         else:
#             xdat[:,:,itim]=read_month(dataset, vardir,var,level,tim,mon[0],'npy',[],verbose=verbose) 

#     return xdat,nlon, nlat,lat,lon,sv

# def read_xarray(dataset='ERA5',region='globe',var='Z',level='500',season='DJF',verbose=False):
#     '''
#     Read npy files from data and generates xarray.

#     This a xarray implementation of read_var. It always grabs the global data.

#     Parameters
#     ----------
#     dataset :   
#         Name of data set   
#     region: 
#         Select region   
#         * *globe*, Entire globe
#         * [East, West, North, South], Specific Region
#     var :   
#          variable name
#     level : 
#         level, either a value or 'SURF' for surface fields

#     season :    
#         Month ('JAN') or season (,'DJF') or annual 'ANN'), or 'ALL' for every year
#     verbose:    
#         True/False -- Tons of Output

#     Returns
#     -------
#     out : xarray   
#         array data 
    
#     '''
    
#     if season != 'ALL':
#         xdat,nlon, nlat,lat,lon,sv=readvar_grid(region='globe',dataset=dataset, \
#                             var=var,level=level,season=season,Celsius=False,verbose=verbose)
#         times = pd.date_range('1979-01-01', periods=40,freq='YS')
#     elif season == 'ALL':
#         xdat,nlon, nlat,lat,lon,sv=readvar_year(region='globe',dataset=dataset, \
#                             var=var,level=level,period='all',Celsius=False,verbose=verbose)
#         times = pd.date_range('1979-01-01', periods=480,freq='MS')
    
#     out = xr.DataArray(xdat, coords=[lat, lon, times], dims=['lat','lon','time'])
#     if sv:
#         out=xr.where(out == sv, np.nan, out)
#     if region != 'globe':
#         out = out.sel(lon = slice(region[0],region[1]), lat = slice(region[2],region[3]))

    
#     return out

# def read_dataset(dataset='ERA5',region='globe',var='Z',level='500',season='DJF',verbose=False):
#     '''
#     Similar to `read_xarray` but returns a ``xarray DataSet``
    
#     '''
    
#     out = read_xarray(dataset=dataset, region=region, \
#                             var=var,level=level,season=season,verbose=verbose)
#     ds = xr.Dataset({var: out})
#     return ds

# def readvar_year(region='globe',period='all',dataset='ERA5',var='Z',level='500',
#                  Celsius=False,verbose=False):
#     """
#     Read Variable from data banks, all month, no averaging
    
#     Parameters
#     ----------
#     Region :    
#         'globe' for global maps, or [east, west, north, south]
#         for limited region, longitude 0-360
#     dataset :   
#          name of data set
#     var :   
#          variable name
#     level : 
#         level, either a value or 'SURF' for surface fields

#     period :    
#         Time period to be read  
#             * 'all' Every time level in databank  
#             * [start_year,end_year] period in those years
     
#     Celsius :   
#         True/False for temperature transform to Celsius
    
#     verbose :   
#         True/False -- tons of output

#     Returns
#     -------

#     xdat : numpy    
#         array data 
#     nlon :  
#         Number of longitudes
#     nlat :  
#         Number of Latitudes
#     lat :   
#         Latitudes
#     lon :   
#         Longitudes

#     Examples
#     --------

#     >>> readvar_year(region='globe',dataset='ERA5',var='Z',level='500',season='JAN',Celsius=False,verbose=False)
#     >>> readvar_year(region='globe',dataset='ERA5',var='SST',level='SURF',season='JAN',Celsius=True,verbose=False)
#     """
    
#     vardir = '.'

#     dat=date_param()

#     grid = DataGrid()
#     nlat = grid[dataset]['nlat']
#     nlon = grid[dataset]['nlon']

#     lat = grid[dataset]['latnp']
#     lon = grid[dataset]['lonnp']
#     #Correct for longitude in ERA5
#     if dataset =='ERA5':
#         lon=lon[:-1]
#     sv=None
#     try:
#         sv=grid[dataset]['special_value']
#         if verbose: print('  Using Special Value ---->', sv)
#     except:
#         print('  Special Value not defined for dataset {}'.format(dataset))

#     ys=grid[dataset][var]['start']
#     ye=grid[dataset][var]['end']
#     ys=1979
#     ye=2018
#     lname=grid[dataset][var]['longname']
#     #Choose period
#     if period =='all':
#         nyears= ye-ys
#         years =[i for i in range(ys,ye+1)]
#     else:
#         nyears= period[1]-period[0]
#         years =[i for i in range(period[0],period[1]+1)]

#     factor=grid[dataset][var]['factor']

#     xdat= np.zeros([nlat,nlon,12*(nyears+1)])
   
#     itim=0
#     dat=date_param()
#     mon=dat['ANN']['month_index']
#     if verbose : print(' Reading ' + var + ' from databank ' + dataset)
#     for tim in years:
#         print('Reading year  ', tim)
#         for imon in mon:       
#             if verbose : print('Reading mon  ', imon)
#             xdat[:,:,itim]=read_month(dataset,vardir,var,level,tim,mon[imon-1],'npy',[],verbose=verbose)            
#             if verbose : print('Reading time  ', itim)
#             itim = itim + 1

#     return xdat,nlon, nlat,lat,lon,sv

@d.get_sections(base='read_era5', sections=['Parameters', 'Returns'])
@d.dedent
def read_era5(var,lev,period='JAN',epoch='AFT', loc = ' ',averaging=True,verbose=False):
    '''
    This routine reads monthly data files from monthly ERA5, 
    optionally combining the backward (1950-1979) and current analysis (1979-2019)

    Parameters
    ----------
    var:
        Variable selected: 
            * u: U-velocity  
            * v: V-velocity
            * t: Temperature
            * w: Vertical Velocity
            * q: Specific Humidity
            * sst: Sea Surface Temperature
            * msl: Mean Sea Level Pressure
            * mtnlwrf: mean_top_net_long_wave_radiation_flux
            * mtnswrf: mean_top_net_short_wave_radiation_flux
            * mslhf: mean_surface_latent_heat_flux
            * msshf: mean_surface_sensible_heat_flux
            * msnswrf: mean_surface_net_short_wave_radiation_flux
            * msnlwrf: mean_surface_net_long_wave_radiation_flux
            * tcw: total_column_water  
            * t2m: 2m_Temperature
    lev: 
        pressure level,
        [10,50,100,150, 200,250,300,400, 500,600, 700,850,925, 1000] 
    period:
        Month or season to be selected. For periods across years, i.e. 'DJF' the
        first and last years are dropped.
        Values are month or season labels
        'JFM','AMJ','JAS','OND','DJF','JJA'
        'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
        'ANN'  -- Entire time series
    epoch:
        * 'BCK' -- Only backward period 1950-1978
        * 'AFT' -- Current period 1979 - 2019
        * 'ALL' -- Combine the two
        * 'V5'  -- ERA5 V5 1940-2022
    loc:    
        Location of the root directory for the data set
    averaging:
        True/False If averaged output is desired
    
    Returns
    -------
    data:
        data in xarray format
    
    '''

    #Form file names
    dash = '/'
    udl =  '_'

    if var == 'SST':
        flev = 'SST'
    else:
        flev = lev

    filbck =loc + dash + var.upper() + dash +var+ udl + flev + udl + 'ER5BCK.nc'
    filaft =loc + dash + var.upper() + dash +var+ udl + flev + udl + 'ER5AFT.nc'
    filV5 =loc + dash + var.upper() + dash +var+ udl + flev + udl + 'V5.nc'

    if epoch == 'AFT':
        dat = xr.open_dataset(filaft)
        enddat=pd.to_datetime(["2019-12-01"])[0]
    elif epoch == 'BCK':
        dat = xr.open_dataset(filbck)
        enddat=pd.to_datetime(["1978-12-01"])[0]
    elif epoch == 'ALL':
        dat1 = xr.open_dataset(filaft)
        dat2 = xr.open_dataset(filbck)
        dat = xr.concat([dat2,dat1],dim='time')
        enddat=pd.to_datetime(["2019-12-01"])[0]
    elif epoch == 'V5':
        dat = xr.open_dataset(filV5)
        enddat=pd.to_datetime(["2022-12-01"])[0]
    else:
        SystemError(f'Wrong Period choice')
    

    # If time has another name change it
    if 'valid_time' in dat.dims:
        dat = dat.rename({'valid_time':'time'})
    if verbose:
        print(f'Selected data from {dat.time[0].data} to {dat.time[-1].data} \n')

    if period != 'ANN':
        perlab,fr = decode_period(period)
        dat = dat.sel(time=dat.time.dt.month.isin(perlab))
        #Adjust for period across years
        if period == 'DJF':        
            dat=dat.where(dat.time > dat.time[1], drop=True).where(dat.time < enddat,drop=True)
        #Check for averaging
        if averaging:
            dat = dat.coarsen(time=fr).mean()
        
    #Adjust names
    dat = dat.rename({'longitude':'lon','latitude':'lat'})
    
    return dat

def decode_period(period):
    '''
    Decode Season or month to index

    '''

    season = ['JFM','AMJ','JAS','OND','DJF','JJA']
    months =[ 'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    seanum = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[12,1,2],[6,7,8]]

    if period  in season:
        res = seanum[season.index(period)]
        fr = 3
    elif period  in months:
        res = months.index(period)  + 1 
        fr = 1      

    return res,fr
@d.dedent
def in_data(var,lev,dataset='ERA5',period='JAN',epoch='AFT', loc = ' ',averaging=True,verbose=False, use_cache=True):
    '''
    This is a wrapper around `read_era5`. Using a file cache to avoid
    slow readings. 
    Optionally combining the backward and current analysis

    Parameters
    ----------

    dataset:
        Data set to be read:
        *   'ERA5' -- Monthly mean Reanalysis
        *   'CERES' -- CERES Satellite Radiance Product 
        *   'GPCP' -- GPCP Precipitation

    %(read_era5.parameters)s   
    
    Returns
    -------

    %(read_era5.returns)s

    '''
    if use_cache and verbose and period != 'ANN':
        print('Using cache ..\n')

    # Consider cache only for ERA5
    if dataset != 'ERA5':
        use_cache = False

    #Form file names
    dash = '/'
    udl =  '_'
   
    cdir = loc + dash + 'DATA_CACHE' 
    file = cdir + dash + var.upper() + udl + lev + udl + epoch + udl+ period + udl + str(averaging) + '.nc'
    # Create cache Directory
    try:
        os.mkdir(cdir)
    except FileExistsError:
        if verbose:
            print("Directory " , cdir ,  " Already Exists")

    if use_cache:
        if period != 'ANN':
            try:
                res = xr.open_dataset(file)
                if verbose:
                    print(f'Reading file {file}')
            except:
                if verbose:
                    print(f'Creating file {file}')
                res = read_era5(var,lev,period=period,epoch=epoch, loc = loc,
                                    averaging=averaging,verbose=verbose)
                res.to_netcdf(file)
        else:
            res = read_era5(var,lev,period=period,epoch=epoch, loc = loc,averaging=averaging,verbose=verbose)
    else:
        if dataset == 'ERA5':
            res = read_era5(var,lev,period=period,epoch=epoch, loc = loc,averaging=averaging,verbose=verbose)
        elif dataset == 'CERES':
            res = read_ceres(var,lev,period=period,epoch=epoch, loc = loc,averaging=averaging,verbose=verbose)
        elif dataset == 'GPCP':
            res = read_GPCP(var,lev,period=period,epoch=epoch, loc = loc,averaging=averaging,verbose=verbose)
        else:
            print(f' Wrong choice in `in_data`')

    return res

# def in_zonal(var,avedim=['time','lon'],**kw):
#     '''
#     Read Zonally Averaged Sections
    
#     Examples
#     --------
#     >>> in_zonal(var,period='DJF',epoch='AFT', loc = ' ',averaging=True)
#     '''


#     totlev = [ '10','50','100','150','200','250','300','400','500','600','700','850','925','1000']
#     first_lev= totlev[0]
#     print(f'Treating level  {var} {first_lev}')
#     if avedim:
#         dd = in_data(var,first_lev,**kw).mean(dim=avedim)
#     else:
#         dd = in_data(var,first_lev,**kw)
#     zon=dd.expand_dims('pressure').assign_coords(pressure=[float(totlev[0])])
    
#     totlev.remove(first_lev)

#     for l in totlev:
#         print(f'Treating level  {var} {l}')
#         if avedim:
#             dd = in_data(var,l,**kw).mean(dim=avedim).expand_dims('pressure').assign_coords(pressure=[float(l)])
#         else:
#             dd = in_data(var,l,**kw).expand_dims('pressure').assign_coords(pressure=[float(l)])
#         zon=xr.concat([zon,dd],dim='pressure')
#     return zon

@d.get_sections(base='read_ceres', sections=['Parameters', 'Returns'])
@d.dedent
def read_ceres(var,lev,period='JAN',epoch='AFT', loc = ' ',averaging=True,verbose=False):
    '''
    This routine reads monthly data files from CERES, 
    

    Parameters
    ----------
    var:
        Variable selected: 
            * toa_sw_all_mon                W/m2    TOA Shortwave Flux - All-Sky
            * toa_lw_all_mon                W/m2    TOA Longwave Flux - All-Sky 
            * toa_net_all_mon               W/m2    TOA Net Flux - All-Sky 
            * toa_sw_clr_c_mon              W/m2    TOA Shortwave Flux - Clear-Sky (for cloud-free areas of region)
            * toa_lw_clr_c_mon              W/m2    TOA Longwave Flux - Clear-Sky (for cloud-free areas of region) 
            * toa_net_clr_c_mon             W/m2    TOA Net Flux - Clear-Sky (for cloud-free areas of region) 
            * solar_mon                     W/m2    Incoming Solar Flux 
            * cldarea_total_daynight_mon    %       Cloud Area Fraction - Daytime-and-Nighttime
            * cldpress_total_daynight_mon   hPa     Cloud Effective Pressure - Daytime-and-Nighttime
            * cldtemp_total_daynight_mon    K       Cloud Effective Temperature - Daytime-and-Nighttime
            * cldtau_total_day_mon          1       Cloud Visible Optical Depth - Daytime 
    lev: 
        TOA
    period:
        Month or season to be selected. For periods across years, i.e. 'DJF' the
        first and last years are dropped.
        Values are month or season labels
        'JFM','AMJ','JAS','OND','DJF','JJA'
        'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
        'ANN'  -- Entire time series
    epoch:
        * 'ALL' -- Entire data series
    loc:    
        Location of the root directory for the data set
    averaging:
        True/False If averaged output is desired
    
    Returns
    -------
    data:
        data in xarray format
    
    '''

    #Form file names
    dash = '/'
    udl =  '_'

    fil =loc + dash + 'CERES' + dash + 'CERES_EBAF-TOA_Ed4.1_Subset_200003-202012.nc'

    if epoch == 'ALL':
        dat = xr.open_dataset(fil)
        enddat=pd.to_datetime(["2020-12-01"])[0]
    else:
        SystemError(f'Wrong Period choice')
    if verbose:
        print(f'Selected data from {dat.time[0].data} to {dat.time[-1].data} \n')


    if period != 'ANN':
        perlab,fr = decode_period(period)
        dat = dat.sel(time=dat.time.dt.month.isin(perlab))
        #Adjust for period across years
        if period == 'DJF':        
            dat=dat.where(dat.time > pd.to_datetime(["2001-03-01"])[0], drop=True).where(dat.time < enddat,drop=True)
        #Check for averaging
        if averaging:
            dat = dat.coarsen(time=fr).mean()
        

    return dat[var]
def read_GPCP(var,lev,period='JAN',epoch='AFT', loc = ' ',averaging=True,verbose=False):
    '''
    This routine reads monthly data files from GPCP
    

    Parameters
    ----------
    var:
        precipitation
    lev: 
        Surface
    period:
        Month or season to be selected. For periods across years, i.e. 'DJF' the
        first and last years are dropped.
        Values are month or season labels
        'JFM','AMJ','JAS','OND','DJF','JJA'
        'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
        'ANN'  -- Entire time series
    epoch:
        * 'ALL' -- Entire data series
    loc:    
        Location of the root directory for the data set
    averaging:
        True/False If averaged output is desired
    
    Returns
    -------
    data:
        data in xarray format
    
    '''
    
    fil = loc +'/Dropbox (CMCC)/DATA/GPCP/precip.mon.mean.nc'

    if epoch == 'ALL':
        dat = xr.open_dataset(fil).precip
        startdat =  dat.time[0]
        enddat =  dat.time[-1]
    else:
        raise SystemError(f'Wrong Period choice')
    if verbose:
        print(f'Selected data from {dat.time[0].data} to {dat.time[-1].data} \n')


    if period != 'ANN':
        perlab,fr = decode_period(period)
         #Adjust for period across years
        if period == 'DJF':        
            dat=dat.where(dat.time > dat.time[10].data, drop=True).where(dat.time < enddat,drop=True)
        dat = dat.sel(time=dat.time.dt.month.isin(perlab))
        #Check for averaging
        if averaging:
            dat = dat.coarsen(time=fr).mean()
        

    return dat
def select_time(dataset_dates,select=None, period=None):
    '''
    Function to create the time range for the data selection.
    A subperiod can be selected within the data set using `select` and `period`,
    `select` is the data set interval , `period` is the calendar choice within the data set. 
    If both are 'none' the entire data set is selected from `dataset_dates` is chosen.

    Parameters
    ----------

    dataset_dates : list
        List with the starting and ending dates of the data set
    select : str
        Selection of the period within the data set given by `dataset_dates`
            * ERA5 -- 1940-2022
            * COBE -- 1891-2020
            * XXSEC -- 1900-2020
    period : str
        Period to be selected
            * JAN -- January
            * JUL -- July
            * ANN -- Annual   
    '''
    if select is not None:
        if select == 'ERA5':
            tstart = 'MON/1/1940'
            tend   = 'MON/12/2022'
        elif select == 'COBE':
            tstart = 'MON/1/1891'
            tend   = 'MON/1/2020'
        elif select == 'XXSEC':
            tstart = 'MON/1/1900'
            tend   = 'MON/1/2020'
        else:
            ValueError(f'Wrong selection {selection} in `select_time`')

    if period is not None:
        if period =='JAN':
            sel_start = tstart.replace('MON','1')
            sel_end = tend.replace('MON','2')
            sel_time = pd.date_range(start=sel_start, end=sel_end, freq='12MS')
        elif period == 'JUL':
            sel_start = tstart.replace('MON','7')
            sel_end = tend.replace('MON','8')
            sel_time = pd.date_range(start=sel_start, end=sel_end, freq='12MS')
        elif period == 'ANN':
            sel_start = tstart.replace('MON','1')
            sel_end = tend.replace('MON','12')
            sel_time = pd.date_range(start=sel_start, end=sel_end, freq='12MS')
        else:
             ValueError(f'Wrong period {period} in `select_time`')

    dstart = dataset_dates[0]
    dtend = dataset_dates[1]
    data_time = pd.date_range(start=dstart, end=dtend, freq='1MS')

    return  (data_time, sel_time) if select is not None  else data_time


        