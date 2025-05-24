import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import math
import numpy as np

import scipy.linalg as sc
import scipy.special as sp
import scipy.interpolate as spint
import scipy.signal as sig
import scipy.spatial.qhull as qhull

import random

import numpy.linalg as lin
import xarray as xr
import pandas as pd

import scipy.ndimage as ndimage

import zapata.lib as lib
import zapata.data as era
import klus.kernels as ker

import tqdm as tm
import mpl_toolkits.axes_grid1 as tl

def zonal_var(dataset,var,season,option='LonTime',verbose=False):
    """
    A routine to average xarray 
    
    This routine will accept xarray up to four dimensions (lat,lon,pressure, time) and return the averaged arrays with compatible dimensions.
    
    Parameters
    ----------
    dataset :      
        Name of the dataset, ``ERA5``, ``GPCP``       
    var :   
        Variable
    season :     
        Month or Season. Resolved from `dat_param`
    option :       
        Control Averaging   
            -  None        No Averaging   
            - 'LonTime'    Longitude and Time   
            - 'Lon'        Longitude    
            - 'Time'       Time averaging   
    verbose:    
        Tons of Output
    
    Returns
    --------
    
    average :
        Average. The dimension is depending on the averaging option chosen
    
    Examples
    --------
    
    >>> zonal_var('ERA5','Z','JAN',option='LonTime')   #Longitude and Time Average for Z from ERA5
    >>> zonal_var('GPCP','TPREP','DJF',option='Time',verbose=True)   # Time average 
    """

    info=era.DataGrid()
    lev=info[dataset][var]['level']
    nlev=len(lev)
    xx=era.read_xarray(dataset='ERA5',var=var,level=str(lev[0]),season=season,verbose=verbose)
    
    if option == 'LonTime':
        zon=xr.DataArray.expand_dims(xx.mean(dim='lon').mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on longitude and time ')
    elif option == 'Time':
        zon=xr.DataArray.expand_dims(xx.mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on  time ')
    elif option == 'Lon':
        zon=xr.DataArray.expand_dims(xx.mean(dim='lon'),dim='pressure').assign_coords(pressure=[lev[0]])
        print(' Averaging on longitude ')
    else:
        zon=xr.DataArray.expand_dims(xx,dim='pressure').assign_coords(pressure=[lev[0]])

    for i in tm.tnrange(1,nlev):
        xx=era.read_xarray(dataset='ERA5',var=var,level=str(lev[i]),season=season)
        if option == 'LonTime':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='lon').mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[i]])
        elif option == 'Time':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='time'),dim='pressure').assign_coords(pressure=[lev[i]])
        elif option == 'Lon':
            xx1=xr.DataArray.expand_dims(xx.mean(dim='lon'),dim='pressure').assign_coords(pressure=[lev[i]])
        else:
            xx1=xr.DataArray.expand_dims(xx,dim='pressure').assign_coords(pressure=[lev[i]])       
        zon=xr.concat([zon,xx1],dim='pressure')
        
    return zon

def smooth_xarray(X,sigma=5,order=0,mode='wrap'):
    """
    Smooth xarray X with a gaussian filter . 

    It uses a routine from scipy ndimage ( ``ndimage.gaussian_filter``). 
    The filter is applied to all dimensions.
    See the doc page of ( ``ndimage.gaussian_filter``) for a full documentation.
    The filter can be used for periodic fields, then the correct setting of `mode` is 'wrap'

    Parameters
    -----------
    X :  
        Input Xarray    
    sigma:  
        Standard deviation for the Gaussian kernel
    order:  
        Order of the smoothing, 0 is a simple convolution
    mode:   
        *The mode parameter determines how the input array is extended when the filter overlaps a border. 
        By passing a sequence of modes with length equal to the number of dimensions of the input array, 
        different modes can be specified along each axis.
        Default value is ‘reflect’.*

        The valid values and their behaviors are as follows:

        *   ‘reflect’ (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last pixel.
        
        *   ‘constant’ (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        
        *   ‘nearest’ (a a a a | a b c d | d d d d)
            The input is extended by replicating the last pixel.

        *   ‘mirror’ (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last pixel.

        *   ‘wrap’ (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.

    Returns
    --------
    smooth_array:   
        numpy array

    Examples
    --------
    Smooth a X[lat,lon] array with nearest repetition in *lat* and periodicity in *lon*
    
    >>> smooth_array(X,sigma=5,order=0,mode=['nearest','wrap']) 
    
    """
    lat = X.lat
    lon = X.lon
    if (X.isnull()).any():
        #there are NaN
        # TO DO
        temp = ndimage.gaussian_filter(X, sigma=sigma, order=order,mode=mode)
    else:
        temp = ndimage.gaussian_filter(X, sigma=sigma, order=order,mode=mode)
    
    zarray = xr.DataArray(temp,dims=('lat','lon'),coords={'lat':lat,'lon':lon})
    return zarray

def anomaly(var,option='anom',freq='month'):
    """
    Compute Anomalies according to *option*

    Long description here.

    Parameters
    ----------
    var :   xarray
        array to compute anomalies
    option :
        Option controlling the type of anomaly calculation  
            =============     ==========================================================
            deviation         Subtract the time mean of the time series
            deviation_std     Subtract the time mean and normalize by standard deviation
            anom              Compute anomalies from monthly climatology    
            anomstd           Compute standardized anomalies from monthly climatology
            None              Return unchanged data
            =============     ==========================================================
    freq :  
        Frequency of data   

    Returns
    -------
    anom :  xarray

    """
    # Correction for very small standard deviations
    TINY = 1.0e-14

    frequency = 'time.' + freq
    if option == 'deviation':
        anom = var - var.mean(dim='time')
    elif option == 'deviation_std':
        anom = (var - var.mean(dim='time'))/var.std(dim='time')
    elif option == 'anom':
        clim = var.groupby(frequency).mean("time")
        anom = var.groupby(frequency) - clim
    elif option == 'anomstd':
        clim = var.groupby(frequency).mean("time")
        climstd = var.groupby(frequency).std("time")
        anom = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                var.groupby(frequency),
                clim,
                climstd + TINY)
    else:
        print(f'No option in anomaly return unchanged data')
        anom=var

    return anom
class Xmat():
    """ This class creates xarrays in vector mathematical form.

    The xarray is stacked along `dims` dimensions
    with the spatial values as column vectors and time as the 
    number of columns.

    Specifying the parameter `option` as `DropNaN` will drop all the NaN values
    and the matrix can then be recontructed using the `expand` method.
    

    Parameters
    ----------
    X : xarray
        `xarray` of at leasts two dimensions
    dims : 
        Dimensions to be stacked, *Default ('lat','lon')*
    options :
        Options for Xmat creation
        None :  Keep NaN (Default)
        DropNaN : Drop NaN values

    Attributes
    ----------
    A : xarray
        Stacked matrix of type *xarray*
    _ntime  :
        Length of time points
    _npoints :
        Length of spatial points
    _F : xarray
        Original matrix of type *xarray* with only NaN values 
    
    Examples    
    --------    
    Create a stacked data matrix along the 'lon' 'lat'  dimension

    >>> Z = Xmat(X, dims=('lat','lon'))

    """

    __slots__ = ('A','_F','_ntime','_npoints','_opt')

    def __init__(
        self,
        X,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        option=None,
        ):

        if not dims:
            SystemError('Xmat needs some dimensions')
            
        self._F = xr.full_like(X,fill_value=np.nan)
        self._opt = option
        self.A = X.stack(z=dims).transpose()
        self._ntime = len(X.time.data)
        self._npoints = len(X.stack(z=dims).z.data)
        print('Created data Matrix X, stacked along dimensions {} '.format(dims))
        if self._opt == 'DropNaN':
            self.A = self.A.dropna(dim='z')
            print(' Creating Matrix with Drop NaN values')
           

    def __call__(self, v ):
        ''' Matrix vector evaluation.'''
        f = self.a @ v
        return f

    def __repr__(self):
        '''  Printing Information '''
        print(f' \n Math Data Matrix \n {self.A}\n')
        print(f' Shape of A numpy array {self.A.shape}')
        return  '\n'
    
    def expand(self):
        '''
        Unroll Xmat matrix to xarray

        Examples    
        --------    
        Unroll a stacked and NaN-dropped matrix `X`

        >>> Xlatlon = X.expand()
        '''

        if self._opt == 'DropNaN':
            Aloc =  self.A.unstack()
            self._F.loc[{'lat':Aloc.lat,'lon':Aloc.lon}] = Aloc
            return self._F
        else:
            return self.A.unstack() 
     
    def svd(self, N=10):
        '''Compute SVD of Data Matrix A.
        
        The calculation is done in a way that the modes are equivalent to EOF

        Parameters
        ----------
        N :  
            Number of modes desired.     
            If it is larger than the number of `time` levels    
            then it is set to the maximum

        Returns
        -------
        out : dictionary
            Dictionary including 
                =================     ==================  
                Pattern               EOF patterns    
                Singular_Values       Singular Values 
                Coefficient           Time Coefficients   
                Varex                 Variance Explained  
                =================     ==================
        Examples         
        --------     
        >>> out = Z.svd(N=10) 

        '''
        #Limit to maximum modes to time levels
        Neig = np.min([N,self._ntime])
        print(f'Computing {Neig} Modes')
        # Prepare arrays
        len_modes = self._ntime
        u = self.A.isel(time=range(Neig)).rename({'time': 'Modes'}).assign_coords(Modes= range(Neig))
        u.name = 'Modes'
        
        #Compute modes
        _u,_s,_v=sc.svd(self.A,full_matrices=False)
    
        #EOF Patterns
        u.data = _u[:,0:Neig]
        #Singular values
        s = xr.DataArray(_s[0:Neig], dims='Modes',coords=[np.arange(Neig)])
        #Coefficients
        vcoeff = xr.DataArray(_v[0:Neig,:], dims=['Modes','Time'],coords=[np.arange(Neig),self.A.time.data])
        # Compute variance explained
        _varex = _s**2/sum(_s**2)
        varex = xr.DataArray(_varex[0:Neig], dims='Modes',coords=[np.arange(Neig)])

        #Output
        out = xr.Dataset({'Pattern':u,'Singular_Values': s, 'Coefficient': vcoeff, 'Varex': varex})
        return out

    def corr(self,y, Dim =('time') , option = None):
        """
        Compute correlation of data matrix `A` with index `y`.

        This method compute the correlation of the data matrix
        with an index of the same length of the `time` dimension of `A`

        The p-value returned by `corr` is a two-sided p-value.  For a
        given sample with correlation coefficient r, the p-value is
        the probability that the absolute value of the  correlation of a random sample x' and y' drawn from
        the population with zero correlation would be greater than or equal
        to the computed correlation. The algorithms is taken from scipy.stats.pearsonsr' that can be consulted for full reference

        Parameters
        ----------
        y : xarray  
            Index, should have the same dimension length `time` 

        option : str
            * 'probability' _Returns the probability (p-value) that the correlation is smaller than a random sample
            * 'signicance'  _Returns the significance level ( 1 - p-value)
        
        Returns
        -------
        According to `option`   

        * None  
            corr :  Correlation array  

        * 'Probability'     
            corr :  Correlation array   
            prob :  p-value array  

        * 'Significance'    
            corr :  Correlation array   
            prob :  Significance array  
        
        Examples
        --------
        Correlation of data matrix `Z` with `index`

        >>> corr = Z.corr(index)
        >>> corr,p = Z.corr(index,'Probability')
        >>> corr,s = Z.corr(index,'Significance')
        """
        index= y - y.mean(dim=Dim)
        _corr = (self.A - self.A.mean(dim=Dim)).dot(index)/    \
               (self.A.std(dim=Dim) * y.std(dim=Dim))/self._ntime

        # The p-value can be computed as
        #     p = 2*dist.cdf(-abs(r))
        # where dist is the beta distribution on [-1, 1] with shape parameters
        # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
        # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
        # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
        # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  

        if option == 'Probability':
            ab = self._ntime/2 - 1
        # Avoid small numerical errors in the correlation
            _p = np.maximum(np.minimum(_corr.data, 1.0), -1.0)
            p = 2*sp.btdtr(ab, ab, 0.5*(1 - abs(_p)))
        
            prob = self.A.isel(time=0).copy()
            prob.data = p
            return _corr , prob
        elif option == 'Significance':
            ab = self._ntime/2 - 1
        # Avoid small numerical errors in the correlation
            _p = np.maximum(np.minimum(_corr.data, 1.0), -1.0)
            p = 2*sp.btdtr(ab, ab, 0.5*(1 - abs(_p)))
        
            prob = self.A.isel(time=0).copy()
            prob.data = 1. - p
            return _corr , prob
        else:
        # return only correlation
            return _corr
    
    def cov(self,y, Dim =('time') ):
        """
        Compute covariance of data matrix `A` with `index`.

        This method compute the correlation of the data matrix
        with an index of the same length of the `time` dimension of `A`

        Examples
        --------
        Covariance of data matrix `Z` with `index`

        >>> cov = Z.cov(index)

        """
        index= (y - y.mean(dim=Dim))
        _cov = (self.A - self.A.mean(dim=Dim)).dot(index)/self._ntime
        return _cov

    def anom(self,**kw):
        """ 
        Creates anomalies.

        This is using the function `anomaly` from `zapata.computation` 
        
        """

        self.A = anomaly(self.A,**kw)
        return

    def detrend(self,**kw):
        '''
        Detrend data using the function scipy.signal.detrend
        '''
        print(f'Detrending data with options -->  {kw}')
        self.A.data = sig.detrend(self.A.data,**kw)
        return



def feature_to_input(k,num,PsiX,Proj,icstart=0.15):
    ''' Transform from Feature space to input space.

    It computes an approximate back-image for the Gaussian kernels and
    and exact backimage for kernels based on scalar product whose nonlinear
    map can be inverted.

    Still working on.

    Parameters  
    ----------  

        k :    Kernel    
            Kernel to be used   
        num :   
            Number of RKHS vectors to transform 
        PsiX :  array (npoints,ntime)
            Original data defininf the kernel   
        Proj :  
            Projction coefficients on Feature Space
        icstart :   
            Starting Value for iteration    
    Returns
    ------- 
        back_image : array(npoints, num)    
    '''
    nx,nt=PsiX.shape
    if nx > 10000 :
        print(' Vector is too large for back-image')
        return
    
    name = k.name
    if name == 'Gaussian':
       
        # Eigenfunctions in the input space
        
        #Expand the eigenfunction in the data space
        # Use iteration by Scholkopf 1999
        xold = np.zeros(PsiX[:,0].shape)
        DataEig=np.zeros([nx,num],dtype=complex)
        for it in range(num):    
            xold[:]=icstart
            vec=Proj[:,it]
            conv=1
            kount=0
            while conv > 1.e-5:
                nom=0.0
                den=0.0        
                for j in range(nt):
                    pr=vec[j]*k(xold,PsiX[:,j])
                    nom=pr*PsiX[:,j] + nom
                    den=pr + den
                xnew=nom/den
                conv = sc.norm(xnew - xold,ord=2)
                #print( '  Convergence/j  ', conv,it)
                xold=xnew  
                kount = kount + 1
                if kount > 1000:
                    print( '  Not Converged -- Convergence/j  ', conv,it,kount,den)
                    break
            DataEig[:,it]= xnew
            print( '  Convergence/num  ', conv,it)
    elif name == 'Polynomial':   
        #Exact method
        print(' Kernel  ',name)
        print('Reconstructing Projection as (nx,nt) ', '(',nx,',',nt,')')
        I = np.eye(nx)
        xold = np.zeros(PsiX[:,0].shape)
        DataEig=np.zeros([nx,num],dtype=complex)
        G00 = ker.gramian2(PsiX,I,k)
        for it in range(num):
            print(' Vector Number  ---> ',it)
            acc = G00 @ Proj
            print(acc.shape,G00.shape,Proj.shape)
            
            #for i in range(nx):
            #    sum=0.0
             #   for j in range(nt):
             #       sum = Proj[j,it]* k(PsiX[:,j],I[:,i])+sum
            DataEig[:,it] = (acc-k.c)**(1/k.p)
    else:
        print('Error in Reconstruction')
    return DataEig   
      
def make_random_index(dskm,inda,X,arealat,arealon):
    ''' Generate an index from random sampling of modes

    It computes an index defined over an area using a 
    random sampling of the modes

    Still working on.

    Parameters  
    ----------  

        dskm:   Data Set   
            Data set containing the Koopman decomposition   
        inda:   
            Number of modes to be considered in the random reconstruction
        X:
            Data array with geographiucal information
        arealat:
            Latitudinal boundaries of index calculation
        arealon:
            Longitudinal boundaries of index calculation
          
    Returns
    ------- 

        c:
            Correlation coefficient
    '''
    iran = random.sample(list(np.arange(len(dskm.eigfun_s.modes))), int(len(inda)))
    mran=(dskm.eigfun_s.isel(modes=iran)@dskm.X_KM_s.isel(modes=iran)).T
    xran = xr.full_like(X.A,0).sel(time=X.A.time[:-1])
    xran.data=2*mran.real
    kran=xran.unstack().sel(lat=slice(arealat[0],arealat[1]),lon=slice(arealon[0],arealon[1])).mean(dim=('lat','lon'))
    # plt.plot(kran)
    return kran
    