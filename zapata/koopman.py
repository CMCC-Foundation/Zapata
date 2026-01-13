'''
Koopman Analysis 
================

Routines and methods for Koopman analysis

This module contains the tools for performing a Koopman or Perron decomposition on
time series of vector data. It is based on Klus routines in module `Klus`.

Classes 
-------
| **Koop**      Transfer Operator
| **KoopGen**   Generator Transfer Operator

Examples
--------

>>> options = {'operator': 'Koopman', 
>>>            'kernel_choice': 'poly', 'k_shift':0.1,
>>>           'poly_shift':0,'poly_order':1,
>>>           'bandwidth': 'std',
>>>           'epsilon':1e-5,'time_data': X.A.time.data[:-1]}
>>> KK=zkop.Koop(vdat,**options)
>>> KK.fit(bandwidth='std')
'''

from distutils import file_util
import math
from tkinter.tix import Select
# 
import numpy as np
import xarray as xr
from sklearn.neighbors import KernelDensity

import klus.algorithms as al
import klus.kernels as kernels
import klus.tools as tools

import scipy.linalg as sc
import numpy.linalg as lin
from scipy.spatial import distance
# 

import tabulate as tab

def choose_eig(option,ww,tol,dt):
    '''
    Choose stable eigenvalues, either according their absolute value
    or according the sign of the real part
    
    Parameters
    ----------

    ww: complex
        Koopman Eigenvalues
    option:
        * `one`       Eigenvalues close to one in Abs less than `tol`
        * `stable`    Real part < 0
        * `unstable`  Real part > 0
    tol:
        Tolerance
    dt:
        Time interval used for the Koopman eigenvalues calculation

    Returns
    -------
    
    ind:
        Index of selected eigenvalues
    '''
    print(f'Choose eigenvalues according to option {option}')
    if option == 'one':    
        ind  = np.where(np.abs(1-np.abs(ww)) < tol)[0]
    elif option == 'stable':
        wlog = np.log(ww)/dt
        ind= np.where(wlog.real < tol )[0]
    elif option == 'unstable':
        wlog = np.log(ww)/dt
        ind = np.where(wlog.real > tol )[0]
    else:
        raise SystemExit(f'Wrong option {option} in Choose_eig')
    return ind

def reconstruct_koopman(w,phi0,v,dt,n,decode=None):
    '''
    
    Linear decomposition with Koopman modes
    
    Parameters
    ----------
    
    w:
        Koopman Generator Eigenvalues
        
    phi0:
        Initial values for the eigenfunctions
    
    v:
        Koopman modes
        
    dt:
        Timesteps
    n:
        Number of time steps
    
    decode: 
        If not empty contains the array for decoding
        to physical space
        
    Returns
    -------
    
    x:
        Predicted values
        
    '''
    
    
    #Choose only eigenfunctions close to 1 and dimensions
    time_range = np.arange(n)
    Ntime = n
   
    try:
        Nmode = len(w)
    except:
        Nmode = 1
      
    #Time Matrix
    D = np.zeros([Nmode,Ntime],dtype='complex')
    
    for i in range(Ntime):
    
        D[:,i] = np.exp(w*i*dt)
 
    try:
        Dphi = np.diag(phi0)
    except:
        Dphi = phi0
        
    x = v.T@Dphi@D
    
    if  decode is None:
        y = x
    else:
        y = decode@x

    return  y.real

def check_sim(ax,X,tit,option='std'):
    '''
    Compute scale for kernel

    Parameters
    ----------

    ax :
        Axes for histogram plot
    X :
        Data Matrix
    tit :
        Title for histogram
    option:
        Type of scale
        *   'std'   Standard deviation
        *   'median'    Median
    
    Return
    ------
    scale : float
        Scale required for the bandwidth
    '''
    
    similarity=distance.squareform(distance.pdist(X.T ,'sqeuclidean'))
    
    if option == 'std':
        scale=np.std(similarity.flatten())
    elif option == 'median':
        scale = np.median(similarity.flatten())
    else:
        raise SystemExit(f'Wrong option {option} in check_sim')
    
    sigma = np.sqrt(scale/2)

    print(f'Bandwidth for normalized distances: sigma {sigma}, Option: {option}, Value = {scale}')
    check  = np.std(distance.squareform(distance.pdist(X.T ,'sqeuclidean'))/(2*sigma**2))
    print(f'Check std for normalized distances {check}')

    ax.hist(similarity.flatten(),200,density=True)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title(tit)
    delta = sigma
    return delta

def order_w(w,option='magnitude',direction='up',tdelta=1):
    '''
    Order Eigenvalues according to `option`

    Parameters
    ----------
    option : String
       * 'magnitude'         abs(w)
       * 'frequency'         w.imag
       * 'growth'            log(w).real/dt
       * 'ones'              abs(w) closest to 1.0
       * 'stable'            log(w).real/dt irrespextive of sign 
    
    direction : String
        * 'up'               descending
        * 'down'             ascending
    
    delta
        Sampling frequency
    
    Returns
    -------

    w0 : Array
        Ordered Koopman Eigenvalues
    w1 : Array
        Ordered generator eigenvalues
    ind : Array
        Index of order for ordering Koopman modes
    '''
    print(' Ordering Eigenvalues as ', option, ' with direction ',direction)
    w_cont = np.log(w)/tdelta
    if option == 'magnitude':
        ind=abs(w).argsort()
    elif option == 'frequency':
        ind=abs(w_cont.imag).argsort()
    elif option == 'growth':
        ind=w_cont.real.argsort()
    elif option == 'one':
        ind=np.abs(np.abs(w) - 1.0).argsort() 
    elif option == 'stable':
        ind=abs(w_cont.real).argsort() 
    else:
        print(' Error in order_w', option, direction)
    
    # Choose direction
    if direction == 'up':
        indu=ind[::-1]
    else:
        indu=ind

    w0=w[indu]
    w1=w_cont[indu]
         
    return w0,w1,indu

def expand_modes(v_o,KM_o,VM_o, vw_o,v_s,KM_s,VM_s,vw_s , \
        modetype='', description='', \
        X=None,file=None):
        '''
        Expand modes into the original data shape, 
        optionally write to disk. complex numbers are not supported by netcdf4, 
        so we use h5netcdf with
        `ds.to_netcdf(file,engine="h5netcdf", invalid_netcdf=True)`

        Parameters
        ----------
        v_o,KM_o,VM_o, vw_o: Arrays
            Original norms, Koopman Modes, Eigenfunction values, Eigenvalues
        v_s,KM_s,VM_s,vw_s: Arrays
            Ordered, norms, Koopman Modes, Eigenfunction values, Eigenvalues
        X: Xmat class
            Original data array with geographical information.
        file: str
            Name of file to be written. Optional, if `None` no file is written
           
        Returns
        -------
        ds : Xarray Dataset
            Dataset with Koopman decomposition

        Examples
        --------
        >>> dskm = zkop.expand_modes(v_o,KM_o,VM_o, vw_o,v_s,KM_s,VM_s,vw_s , X=X,file='filename')
        >>> # Read with
        >>> dds=xr.open_dataset('filename',engine='h5netcdf')
        '''
        
        
        X_KM_o = xr.full_like(X.A,0,dtype='float').isel(time=slice(0,KM_o.shape[1])).rename({'time':'modes'})
        X_KM_o.data = KM_o
        X_KM_s = xr.full_like(X.A,0,dtype='float').isel(time=slice(0,KM_s.shape[1])).rename({'time':'modes'})
        X_KM_s.data = KM_s

        # Create Data Sets for Koopman modes
        ds = xr.Dataset(
            data_vars=dict(
            X_KM_o=(["x", "modes"], X_KM_o.data),
            X_KM_s=(["x", "modes"], X_KM_s.data),
            eigfun_o=(["time", "modes"], VM_o.data),
            eigfun_s=(["time", "modes"], VM_s.data),
            eigval_s=(["modes"],  vw_s.data ),
            eigval_o=(["modes"],  vw_o.data ),
            vnorm_o= (["modes"],  v_o.data ), 
            vnorm_s= (["modes"],  v_s.data ),                     ),

        coords=dict(
            feature=(["x"], np.arange(KM_o.shape[0])),
            time=(["time"], np.arange(VM_o.shape[0])),
            modes=(["modes"], np.arange(KM_o.shape[1]) ) ),
        
        attrs=dict(type=modetype, description=description)  )

        if file:
            ds.to_netcdf(file,engine="h5netcdf", invalid_netcdf=True)

        return ds
def find_modes(tper,wto, vrange, **kwargs):
    '''
    Find modes close to a specific Period `tper`
    within a specified `vrange` in month

    Parameters
    ----------

    tper:
        Target Period (in years)
    wto:
        Koopman Generator Eigenvalues
        Also accept all other arguments for `isclose`
    vrange:
        Range (months)

    Returns
    -------

    ind:
        Index into closest period to `tper`
    '''
    pi2 = 2*math.pi

    vtol = vrange/12
    print(f'Seaching for modes close to {tper} Yr periods, with range {vrange} months')
    near_per = np.where(np.isclose(pi2/abs(wto.imag)/12, tper,atol=vtol, **kwargs))
    for i in near_per:
        print(f'Found modes {i}/{len(near_per[0])},   \tPeriod {pi2/wto[i].imag/12}\n')
    return near_per[0]

def density_estimate(data,interval=[-1,-1,400],kernel='gaussian',bw=0.05,**kwargs):
    '''
    Compute density estimation
    Using kde from sklearn

    Parameters
    ----------

    data: 1D array
        Array 1D of data to be analyzed
    interval: list
        [min,max, number_of_points]
    kernel:
        kernel for estimation (default gaussian)
    bw:
        Bandwidth for the kernel

    Other arguments to `kde` can be passed on

    Returns
    -------

    X : Array
        Points where the distribution is computed
    points : Array
        Value of the distribution
    kde : Object
        The Density object

    Examples
    --------
    >>> density_estimate(data,interval=[-1,-1,400],kernel='gaussian',bw=0.05)

    '''
    Xplot = np.linspace(interval[0],interval[1],interval[2])[:, np.newaxis]
    vmode=np.zeros([len(data),1])
    vmode[:,0] = data
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(vmode)
    log_dens = kde.score_samples(Xplot)
    points= np.exp(log_dens)

    return Xplot[:,0], points, kde

def one_over(y):
    """
    Vectorized 1/x, treating x==0 manually

    """
    x = np.array(y).astype(float)
    near_zero = np.isclose(y, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1. / x[~near_zero]
    return x
def one_over_years(y):
    """Vectorized 1/x, treating x==0 manually
    Period calculation from imaginary part
    of generator eigenvalues 2 pi/log(w.imag)
    """
    x = np.array(y).astype(float)
    near_zero = np.isclose(y, 0)
    x[near_zero] = np.inf
    #Transform in years
    x[~near_zero] = 2*math.pi / x[~near_zero]/12
    return x
# 
class Koop():
    """ This class creates the Koopman operator

    The input array is supposed to be in features/samples form
    The augemnted matrix is created inside

    Parameters
    ----------
    vdat: array 
        Data array with features and samples
    
    **kwargs : dict
        * 'operator'          Choice of operator `Koopman` or `Perron` 
        * 'deltat'            Time interval between samples
        * 'kernel_choice'     Kernel choice , `gauss`,`poly`
        * 'sigma'             Bandwidth for Gaussian kernel
            * 'std'   -- Standard Deviation of distances
            * 'median -- Median of distances
            * '100'   -- Explicit value
        * 'epsilon'             Regularization parameter 
        * 'maxeig'     Max number of eigenvalues to compute
        * 'poly_order': 1     Order of polynomial kernel
        * 'poly_shift':0      Shift of polynomial kernel 

    Attributes
    ----------
    PsiX : numpy array
        Reduced data matrix of type *array*
    PsiY : numpy array
        Shifted data matrix of type *array*
    ntime  : int
        Number of samples or time points
    npoints : int
        Number of features or spatial points
    deltat : float
        Time interval between samples
    operator: str
        Operator chosen {'Koopman', 'Perron'}
    kernel_choice: object
        Kernel used in the estimator
    epsilon : float
        Tikhonov Regularization parameter {1e-5}
    sigma : float
        Bandwidth for Gaussian kernel
    maxeig : int
        Maximum number of eigenvalues to compute
    poly_order: int
        Order polynomial kernel
    poly_shift: float
        Shift polynomial kernel
    ww: array
        Koopman eigenvalues
    wwg: array
        Koopman generator eigenvalues
    cc: array
        Koopman eigenfunctions coefficients
    vv: array
        Koopman eigenfunction values on the samples
    wf: array
        Filtered Koopman eigenvalues
    wfg: array
        Filtered Koopman generator eigenvalues
    vvf: array
        Filtered Koopman eigenfunction values on the samples
    Gxx: array
        Kernel Matrix Gxx
    Gxy: array
        Kernel Matrix Gxy
    ker: object
        Object Kernel used
    time_data:
        Time array (optional)

    Examples    
    --------    
    Create a Koopman operator 

    >>> K = Koop(X, {operator: 'koopman', 
            kernel_choice : 'gauss', sigma : 42, epsilon : 1e-5})

    """

    __slots__ = ('PsiX','PsiY','ntime','npoints', 'operator', 'deltat','time_data',\
                'kernel_choice','epsilon','sigma','maxeig','bandwidth',\
                'poly_order','poly_shift', 'k_shift',\
                'wf','wfg','vvf',\
                'ww','wwg','vv','cc','Gxx','Gxy','ker')
    

    def __init__(
        self,
        vdat,
        **kwargs ):
        
        #Set default arguments
        defaultOptions = {'operator': 'Koopman', 'deltat':1,
            'kernel_choice' : 'gauss', 'bandwidth' : 'std', 'epsilon' : 1e-5, 'maxeig':3000,
            'poly_order': 1,'poly_shift':0,'k_shift':0.0, 'time_data': None}
        options ={**defaultOptions, **kwargs}
        
        self.PsiX = vdat[:,:-1]
        '''Reduced Data matrix of type (`array`)'''
        self.PsiY = vdat[:,1:]
        '''Shifted data matrix of type (`array`)'''
        self.ntime = self.PsiX.shape[1]
        '''Number of samples or time points (`int`)'''
        self.npoints = self.PsiX.shape[0]
        '''Number of features or spatial points (`int`)'''
        self.deltat = 1
        '''Time interval between samples (`float`)'''
        
        self.bandwidth = None
        '''Option Bandwidth for Gaussian kernel (`str`)'''
        self.sigma = None
        '''Bandwidth for Gaussian kernel (`float`)'''
        self.operator = options['operator']
        '''Operator chosen (`str`) {'Koopman', 'Perron'}'''
        self.kernel_choice = options['kernel_choice']
        '''Kernel used in the estimator (`str`)'''
        self.epsilon = options['epsilon']
        '''Tikhonov Regularization parameter (`float`) {1e-5}'''
        self.maxeig = options['maxeig']
        '''Maximum number of eigenvalues to compute (`int`)'''
        self.poly_order = options['poly_order']
        '''Order polynomial kernel (`int`)'''
        self.poly_shift = options['poly_shift']
        '''Shift polynomial kernel (`float`)'''
        self.k_shift = options['k_shift']
        '''Shift  kernel (`float`)'''
        if options['time_data'] is not None:
            self.time_data = options['time_data']
        else:
            self.time_data = np.arange(self.ntime)
        '''Time array (`array`)'''

        print(f'Created {self.operator} Estimator \n for {self.ntime} samples and {self.npoints} features, \
            {self.deltat} time interval' )

        self.wf = None
        '''Filtered Koopman eigenvalues (`array`)'''
        self.wfg = None
        '''Filtered Koopman generator eigenvalues (`array`)'''
        self.vvf = None
        '''Filtered Koopman eigenfunction values on the samples (`array`)'''

        self.ww = None
        '''Koopman eigenvalues (`array`)'''
        self.wwg = None
        '''Koopman generator eigenvalues (`array`)'''
        self.vv = None
        '''Koopman eigenfunction values on the samples (`array`)'''
        self.cc = None
        '''Koopman eigenfunctions coefficients (`array`)'''
        self.Gxx = None
        '''Kernel Matrix Gxx (`array`)'''
        self.Gxy = None
        '''Kernel Matrix Gxy (`array`)'''
        self.ker = None
        '''Kernel used (`object`)'''

    def __call__(self, v ):
        ''' Not implemented'''
        print('Not implemented')
        return 

    def __repr__(self):
        '''  Printing Information '''
        print(f'Koopman Estimator for operator {self.operator} with paramenters:\n \
                Kernel_choice {self.kernel_choice}\n \
                Maxeig {self.maxeig}\n \
                Epsilon {self.epsilon}  \n ')
        if self.kernel_choice == 'poly':
            print(f' Poly_Order {self.poly_order} \n \
                     Poly_Shift {self.poly_shift}\n ')
        elif self.kernel_choice == 'gauss':
            print(f'  Bandwidth {self.sigma}\n \
                      ')
        return '\n'

    
    def fit(self,bandwidth=None,verbose=False):
        ''' 
        Fitting estimator.
        Select bandwidth with `bandwidth`

        Parameters
        ----------
        bandwidth: string
            Select the bandiwidth for the Gaussian kernel
                * 'std'             Use bandwidth that make the distances of std 1
                * 'median'          Use median value of distances
                * '100'             Use this explicit numerical value    
        verbose: Boolean
            True    Print Messages
            False   No Messages  (Defaults)        
        Returns
        -------  
        ww:
            Eigenvalues
        self.wwg = np.log(ww_tot)/self.deltat:
            Generator Eigenvalues
        self.vv:
            Value of Eigenfunctions at samples
        self.Gxx:
            kernel matrix Gxx
        self.Gxy:
            kernel matrix Gxy
        self.ker:
            Kernel used

        '''
    
        
        if self.operator == 'Perron': 
            ops='P'
        elif self.operator =='Koopman':
            ops='K'
        else:
            raise ValueError('TRANSFER_OP_GAUSS:  --- Wrong Operator choice ')
        print(' Calculating {} Operator '.format(self.operator))

        
        self.bandwidth = bandwidth

        if bandwidth is None or bandwidth == 'std':
            similarity=distance.squareform(distance.pdist(self.PsiX.T ,'sqeuclidean'))
            scale=np.std(similarity.flatten())
        elif bandwidth == 'median':
            similarity=distance.squareform(distance.pdist(self.PsiX.T ,'sqeuclidean'))
            scale = np.median(similarity.flatten())
        else:
            scale = 2*(float(bandwidth))**2
    
        self.sigma = np.sqrt(scale/2)

        if self.kernel_choice == 'gauss':
            if verbose:
                print(f'Using option {bandwidth} for a  {scale} and sigma {self.sigma}\n')
            k = kernels.gaussianKernel(self.sigma)
        elif self.kernel_choice == 'poly':
            k = kernels.polynomialKernel(p=self.poly_order,c=self.poly_shift)
        elif self.kernel_choice == 'gaussShifted':
            k = kernels.gaussianKernelShifted(self.sigma, shift=self.k_shift)
        else:
            raise ValueError(f'KK_fit:  --- WrongKernel choice {self.kernel_choice}')
        
        ww_tot,vv_tot,cc_eig,Am,Gxx,Gxy =al.kedmd(self.PsiX,self.PsiY, k, \
                epsilon=self.epsilon, evs=self.maxeig, operator=ops,kind='kernel')
       
        if verbose:
            print(f'Computed Transfer Eigenvalues')
            print(f'Rank of Gxx Matrix {lin.matrix_rank(Gxx)} of dimension {Gxx.shape[0]}\n')
        
        self.ww = ww_tot
        self.wwg = np.log(ww_tot)/self.deltat
        self.vv = vv_tot
        self.cc = cc_eig
        self.Gxx = Gxx
        self.Gxy = Gxy
        self.ker = k

        return 
    
    def order(self,choice='frequency',direction='down',cut_period=3):
        ''' 
        Order Eigenvalues according to `option`

        Parameters
        ----------
        option:
            * 'magnitude'         abs(w)
            * 'frequency'         w.imag
            * 'growth'            log(w).real/dt
            * 'ones'              abs(w) closest to 1.0
            * 'stable'            log(w).real/dt irrespextive of sign 
        
        direction:
            * 'up'               descending
            * 'down'             ascending
        
        cut_period:
            Eliminate modes with shorter periods (sample units)
        
        Returns
        -------

        wf:
            Ordered Koopman Eigenvalues
        wfg:
            Ordered generator eigenvalues
        vvf:
            Ordered Koopman eigenfunctions
            
        '''

        #  cut period in (months)

        if cut_period is not 0:
            wmax = np.max(np.where(2*math.pi/self.wwg.imag > cut_period))
        else:
            wmax = len(self.wwg.imag)

        _wf =self.ww[0:wmax]
        _vvf = self.vv[:,0:wmax]
        print(f'Eliminating frequency higher than {cut_period} months, \
            original modes {len(self.ww)}, remaining {len(_wf)}')
        
        wf,wfg, indf = order_w(_wf,option=choice,direction=direction,tdelta=self.deltat)

        self.wf = wf
        self.wfg = wfg
        self.vvf = _vvf[:,indf]

        return wf, wfg, self.vvf
    
    def eigone(self,ww,vv,tol=0.1,select='one',verbose=True):
    
        '''  
        Choose stable eigenvalues, either according their absolute value
        or according the sign of the real part

        Parameters
        ----------

        ww: complex
            Koopman Eigenvalues
        vv: array complex
            Koopman eigenfunctions
        select:
            * `one`       Eigenvalues close to one in Abs less than `tol`
            * `stable`    Real part < 0
            * `unstable`  Real part > 0
        tol:
            Tolerance
        verbose:
            Print Diagnostics
        

        Returns
        -------

        ww:
            Selected eigenvalues
        vv:
            Selected eigenfunctions
        ind:
            Index of selected eigenvalues
        '''
        
        print(f'Keeping only EI close to unity less than tol {tol}')

        indone = choose_eig(select,ww,tol,self.deltat)
        N1 = len(indone)
        wone = ww[indone]
        print(f'\n Trace of generator eigenvalues close to unity according to tolerance {tol} --->  {sum(np.log(wone))}')
        print(f' Factor of Phase Volume contraction {np.exp(sum(np.log(wone))):.6f} per month\n')
        # Order retained unitary eigenvalues and obtain generator eigenvalues 'expwr'
        print(f'Number of Eigenvalues found {N1} over {len(ww)} for kmode tol {tol}\n\n')
        # _,expwr,indexpwr = order_w(wone,option='stable',direction='down',tdelta=1)

        headr = ('Mode','Real Eig','Im Eig','Abs','Gen Real (Yr)','Gen Imag (Yr)')
        kaz_data = [(i,wone[i].real, wone[i].imag, abs(wone[i]), 1./np.log(wone[i]).real/12,   2*math.pi/np.log(wone[i]).imag/12) for i in range(min(len(wone),50))]
        if verbose:
            print(tab.tabulate(kaz_data, headr,stralign='center',tablefmt='github'))
        _vv = vv[:,indone]

        return wone,_vv,indone

    def compute_modes(self,ww,vv,modetype='',description='',normalization=False):
        '''
        Compute Koopman Modes
        
        Parameters
        ----------
        
        ww:
            Koopman eigenvalues
        vv:
            Value of Koopman EIgenfunctions at samples
        modetype:
            Descriptive label 
        description:
            Data Descritpion
        normalization:
            If `False` (Default) the eigenfunctions are not normalized, se to`True`
            to get Koopman modes over normalized eigenfunctions.
        
        
        Returns
        -------
        
        ds : data set
            .. table:: Content for `ds`
               :widths: auto

               ======================        ============================================
               Variable                      Description
               ======================        ============================================
               keig(time,modes)              Normalized Koopman eignfunction values at sample points
               kmodes(features,modes)        Koopman modes
               periods                       Periods of Koopman modes
               eigenvalues                   Koopman Eigenvalues
               ======================        ============================================
        '''
        
        Nmodes = vv.shape[1]
        Ntimes = self.ntime 
        Nfeatures = self.npoints

        # Normalize Koopman eigenfunctions
        if normalization:
            vvnorm = sc.norm(vv,axis=0)
            vnormalized = vv@np.diag(1./vvnorm)
        else:
            vnormalized = vv

        # The ordering is consistent with Mezic
        Phiinv, vrank = sc.pinv(vnormalized,return_rank=True)
        KMM = Phiinv@(self.PsiX.T)
        print(f'Computed Kmodes {KMM.shape} for {vv.shape} normalized eigenfunctions')
        print(f'Computed Kmodes {Nmodes} {Nfeatures} vv  {Ntimes} {Nmodes} eigenfunctions')
        print(f'Rank of V matrix {vrank}')
        

        # Create Data Sets for Koopman modes
        ds = xr.Dataset(
            data_vars=dict(
            kmodes=(["x", "modes"], KMM.T),
            eigfun=(["time", "modes"], vnormalized),
            periods=(["modes"],  2*math.pi/np.log(ww).imag/12),
            eigval=(["modes"],  ww )                                 ),

        coords=dict(
            feature=(["x"], np.arange(Nfeatures)),
            time=(["time"], self.time_data),
            modes=(["modes"], np.arange(Nmodes) ) ),
        attrs=dict(type=modetype, description=description)  )

        return ds

    def reconstruct_koopman(self,ds,decode=None,select_modes=None):
        '''
    
        Reconstruction of data from Koopman modes
        
        Parameters
        ----------
       
        ds:
            Koopman decomposition from `compute_modes`
        
        decode:
            Array decoding features
            
        decode: 
            If not empty contains the array for decoding
            to physical space
        
        select_modes: list
            If not empty reconstrut using only selected modes
            [2,7,4]
            

            
        Returns
        -------
        
        data : Array
            Reconstructed data
            
        '''

        if select_modes is None:
            y = ds.eigfun@ds.kmodes
        else:
            y = ds.eigfun.isel(modes=select_modes)@ds.kmodes.isel(modes=select_modes)

        #Check Reality
        if np.sum(y.imag**2) > 1.e-14:
             raise ValueError(f'reconstruct_koopman:  --- Array not Real {np.sum(y.imag**2)}')
        else:
            y = y.real

        if decode is None:
            return y
        else:
            return decode@y.data.T
    def find_norm_modes(self,KM, KE, KEI, sort='sort',decode=None,simplify_conjugates=True):
        '''
        Compute norms of eigenfunctions 
        eliminate complex conjugates, optionally sort in order
        of magnitude  

        Parameters
        ----------
        KM:
            Koopman Modes (KM)
        KE: 
            Koopman Eigenfunctions (KE)
        KEI:
            Koopman EIgenvalues (KEI)
        sort:
            'sort' -- get also sorted eigenvector in order of magnitude
        decode:
            None -- No decoding is applied, otherwise use array `decode`
        simplify_conjugates:
            If `True` only modes with imaginary part >= 0 are retained

        Returns
        -------

        (vnorm, KM, VM, KEI):
            Koopman Modes Norms, Koopman Modes, Eigenfunctions, Eigenvalues
        (vnormsort, KM_sort, VM_sort, KEI_sort):
            Sorted Koopman Modes Norms, Koopman Modes, Eigenfunctions, Eigenvalues
        '''

        # Eliminate complex conjugates
        if simplify_conjugates:
            print(f'\n `find_norm_modes` Retain only positive frequencies or zero \n')
            ivar = np.where(KEI.imag >= 0)
            KM_single = KM[:,ivar[0]]
            VM_single = KE[:,ivar[0]]
            wti = KEI[ivar[0]]
        else:
            KM_single = KM[:,:]
            VM_single = KE[:,:]
            wti = KEI[:]

        if decode is None:
            KM_out = KM_single.data
        else:
            KM_out =  decode@KM_single.data

        vvar = np.diag(KM_single.T.conj().data@KM_single.data).real
        vnorm = np.sqrt(vvar)

        if sort == 'sort':
            # Sort in order of magnitude
            indnorm = np.argsort(vnorm)[::-1]
            return (vnorm,KM_out, VM_single, wti), (indnorm, vnorm[indnorm],KM_out[:,indnorm], VM_single[:,indnorm], wti[indnorm])
        else:
            return (vnorm,KM_out, VM_single, wti)
    def eigenfunction_value( self, x, mode ):
        '''
        Compute eigenfunction value at point x
        
        Parameters
        ----------
        
        x:
            Point at which eigenfunction is computed
        mode:
            Eigenfunction number
        
        Returns
        -------
        
        eigenfunction value
        '''
        k = self.ker
        f = 0
        for i in range(self.ntime):
            f +=  self.cc[i,mode]*k(x, self.PsiX[:, i])
        return f

    
        
    
class KoopGen():
    """ This class creates the Generator Koopman operator

    The input array is supposed to be in features/samples form
    The augemnted matrix is created inside

    Parameters
    ----------
    vdat: array 
        Data array with features and samples
    
    **kwargs : dict
        * 'operator'          Choice of operator `Koopman` or `Perron` 
        * 'deltat'            Time interval between samples
        * 'kernel_choice'     Kernel choice , `gauss`,`poly`
        * 'sigma'             Bandwidth for Gaussian kernel
            * 'std'   -- Standard Deviation of distances
            * 'median -- Median of distances
            * '100'   -- Explicit value
        * 'epsilon'             Regularization parameter 
        * 'maxeig'     Max number of eigenvalues to compute
        * 'poly_order': 1     Order of polynomial kernel
        * 'poly_shift':0      Shift of polynomial kernel 

    Attributes
    ----------
    PsiX : numpy array
        Reduced data matrix of type *array*
    PsiY : numpy array
        Shifted data matrix of type *array*
    ntime  : int
        Number of samples or time points
    npoints : int
        Number of features or spatial points
    deltat : float
        Time interval between samples
    operator: str
        Operator chosen {'Koopman', 'Perron'}
    kernel_choice: object
        Kernel used in the estimator
    epsilon : float
        Tikhonov Regularization parameter {1e-5}
    sigma : float
        Bandwidth for Gaussian kernel
    maxeig : int
        Maximum number of eigenvalues to compute
    poly_order: int
        Order polynomial kernel
    poly_shift: float
        Shift polynomial kernel
    ww: array
        Koopman eigenvalues
    wwg: array
        Koopman generator eigenvalues
    vv: array
        Koopman eigenfunction values on the samples
    wf: array
        Filtered Koopman eigenvalues
    wfg: array
        Filtered Koopman generator eigenvalues
    vvf: array
        Filtered Koopman eigenfunction values on the samples
    Gxx: array
        Kernel Matrix Gxx
    Gxy: array
        Kernel Matrix Gxy
    ker: object
        Object Kernel used

    Examples    
    --------    
    Create a Koopman operator 

    >>> K = Koop(X, {operator: 'koopman', 
            kernel_choice : 'gauss', sigma : 42, epsilon : 1e-5})

    """

    __slots__ = ('PsiX','PsiY','ntime','npoints', 'operator', 'deltat',\
                'kernel_choice','epsilon','sigma','maxeig','bandwidth',\
                'poly_order','poly_shift', 'k_shift',\
                'wf','wfg','vvf',\
                'ww','wwg','vv','Gxx','Gxy','ker')
    

    def __init__(
        self,
        vdat,
        **kwargs ):
        
        #Set default arguments
        defaultOptions = {'operator': 'Koopman', 'deltat':1,
            'kernel_choice' : 'gauss', 'bandwidth' : 'std', 'epsilon' : 1e-5, 'maxeig':2000,
            'poly_order': 1,'poly_shift':0,'k_shift':0.0}
        options ={**defaultOptions, **kwargs}
        
        self.PsiX = vdat[:,:-1]
        '''Reduced Data matrix of type (`array`)'''
        self.PsiY = vdat[:,1:]
        '''Shifted data matrix of type (`array`)'''
        self.ntime = self.PsiX.shape[1]
        '''Number of samples or time points (`int`)'''
        self.npoints = self.PsiX.shape[0]
        '''Number of features or spatial points (`int`)'''
        self.deltat = 1
        '''Time interval between samples (`float`)'''
        self.bandwidth = None
        '''Option Bandwidth for Gaussian kernel (`str`)'''
        self.sigma = None
        '''Bandwidth for Gaussian kernel (`float`)'''
        self.operator = options['operator']
        '''Operator chosen (`str`) {'Koopman', 'Perron'}'''
        self.kernel_choice = options['kernel_choice']
        '''Kernel used in the estimator (`str`)'''
        self.epsilon = options['epsilon']
        '''Tikhonov Regularization parameter (`float`) {1e-5}'''
        self.maxeig = options['maxeig']
        '''Maximum number of eigenvalues to compute (`int`)'''
        self.poly_order = options['poly_order']
        '''Order polynomial kernel (`int`)'''
        self.poly_shift = options['poly_shift']
        '''Shift polynomial kernel (`float`)'''
        self.k_shift = options['k_shift']
        '''Shift  kernel (`float`)'''

        print(f'Created {self.operator} Estimator \n for {self.ntime} samples and {self.npoints} features, \
            {self.deltat} time interval' )


        self.wf = None
        '''Filtered Koopman eigenvalues (`array`)'''
        self.wfg = None
        '''Filtered Koopman generator eigenvalues (`array`)'''
        self.vvf = None
        '''Filtered Koopman eigenfunction values on the samples (`array`)'''

        self.ww = None
        '''Koopman eigenvalues (`array`)'''
        self.wwg = None
        '''Koopman generator eigenvalues (`array`)'''
        self.vv = None
        '''Koopman eigenfunction values on the samples (`array`)'''
        self.Gxx = None
        '''Kernel Matrix Gxx (`array`)'''
        self.Gxy = None
        '''Kernel Matrix Gxy (`array`)'''
        self.ker = None
        '''Kernel used (`object`)'''

    def __call__(self, v ):
        ''' Not implemented'''
        print('Not implemented')
        return 

    def __repr__(self):
        '''  Printing Information '''
        print(f'Koopman Estimator for operator {self.operator} with paramenters:\n \
                Kernel_choice {self.kernel_choice}\n \
                Maxeig {self.maxeig}\n \
                Epsilon {self.epsilon}  \n ')
        if self.kernel_choice == 'poly':
            print(f' Poly_Order {self.poly_order} \n \
                     Poly_Shift {self.poly_shift}\n ')
        elif self.kernel_choice == 'gauss':
            print(f'  Bandwidth {self.sigma}\n \
                      ')
        return '\n'

    def fit(self,bandwidth=None,derivative=None):
        ''' 
        Fitting generator estimator.
        Select bandwidth with `bandwidth`

        Parameters
        ----------
        bandwidth: string
            Select the bandiwidth for the Gaussian kernel
                * 'std'             Use bandwidth that make the distances of std 1
                * 'median'          Use median value of distances
                * '100'             Use this explicit numerical value    
        derivative: array
            Derivative of the data to be used in the generator.
            If not provided, the derivative is computed. (not implemented)
                
        Returns
        -------  
        ww:
            Eigenvalues
        self.wwg = np.log(ww_tot)/self.deltat:
            Generator Eigenvalues
        self.vv:
            Value of Eigenfunctions at samples
        self.Gxx:
            kernel matrix Gxx
        self.Gxy:
            kernel matrix Gxy
        self.ker:
            Kernel used

        '''

        
        if self.operator == 'Perron': 
            ops='P'
        elif self.operator =='Koopman':
            ops='K'
        else:
            raise ValueError('TRANSFER_OP_GAUSS:  --- Wrong Operator choice ')
        print(' Calculating {} Operator '.format(self.operator))

        self.bandwidth = bandwidth

        if bandwidth is None or bandwidth == 'std':
            similarity=distance.squareform(distance.pdist(self.PsiX.T ,'sqeuclidean'))
            scale=np.std(similarity.flatten())
        elif bandwidth == 'median':
            similarity=distance.squareform(distance.pdist(self.PsiX.T ,'sqeuclidean'))
            scale = np.median(similarity.flatten())
        else:
            scale = 2*(float(bandwidth))**2
       
        self.sigma = np.sqrt(scale/2)
        print(f'Using bandwidth {bandwidth} for a scale of {scale} and sigma {self.sigma}\n')

        if self.kernel_choice == 'gauss':
            k = kernels.gaussianKernel(self.sigma)
        elif self.kernel_choice == 'poly':
            k = kernels.polynomialKernel(p=self.poly_order,c=self.poly_shift)
        elif self.kernel_choice == 'gaussShifted':
            k = kernels.gaussianKernelShifted(self.sigma, shift=self.k_shift)
        else:
            raise ValueError(f'KK_fit:  --- WrongKernel choice {self.kernel_choice}')
        if derivative is not None:
            self.PsiY = derivative
        else:
            raise ValueError(f'KK_fitgen: Provide value for Derivative {derivative}')

        ww_tot,vv_tot,A,Gxx,Gxy =al.kgedmd(self.PsiX,self.PsiY, k, \
                epsilon=self.epsilon, evs=self.maxeig, operator=ops,kind='kernel')
        
        print(f'Computed Transfer Eigenvalues')
        self.ww = np.exp(ww_tot*self.deltat)
        self.wwg = ww_tot 
        self.vv = vv_tot
        self.Gxx = Gxx
        self.Gxy = Gxy
        self.ker = k

        return 
    
    
    def order(self,choice='frequency',direction='down',cut_period=3):
        ''' 
        Order Eigenvalues according to `option`

        Parameters
        ----------
        option:
            * 'magnitude'         abs(w)
            * 'frequency'         w.imag
            * 'growth'            log(w).real/dt
            * 'ones'              abs(w) closest to 1.0
            * 'stable'            log(w).real/dt irrespextive of sign 
        
        direction:
            * 'up'               descending
            * 'down'             ascending
        
        cut_period:
            Eliminate modes with shorter periods (sample units)
        
        Returns
        -------

        wf:
            Ordered Koopman Eigenvalues
        wfg:
            Ordered generator eigenvalues
        vvf:
            Ordered Koopman eigenfunctions
            
        '''

        #  cut period in (months)

        wmax = np.max(np.where(2*math.pi/self.wwg.imag > cut_period))
        _wf =self.ww[0:wmax]
        _vvf = self.vv[:,0:wmax]
        print(f'Eliminating frequency higher than {cut_period} months, \
            original modes {len(self.ww)}, remaining {len(_wf)}')
        
        wf,wfg, indf = order_w(_wf,option=choice,direction=direction,tdelta=self.deltat)

        self.wf = wf
        self.wfg = wfg
        self.vvf = _vvf[:,indf]

        return wf, wfg, self.vvf
    
    def eigzero(self,ww,vv,tol=0.1,option='zero'):
    
        '''  
        Choose stable eigenvalues, either according their absolute value
        or according the sign of the real part

        Parameters
        ----------

        ww: complex
            Koopman Generator Eigenvalues
        vv: array complex
            Koopman eigenfunctions
        select:
            * `zero`      Eigenvalues close to zero real part less than `tol`
            * `stable`    Real part < 0
            * `unstable`  Real part > 0
        tol:
            Tolerance
        

        Returns
        -------

        ww:
            Selected eigenvalues
        vv:
            Selected eigenfunctions
        ind:
            Index of selected eigenvalues
        '''
        
        print(f'Keeping only EI close to zero less than tol {tol}')

        if option == 'zero':    
            indone  = np.where(np.isclose(ww.real,0,atol=tol))[0]
        elif option == 'stable':
            indone = np.where(ww.real < 0 )[0]
        elif option == 'unstable':
            indone  = np.where(ww.real > 0 )[0]
        else:
            raise SystemExit(f'Wrong option {option} in Choose_eig')
        
        N1 = len(indone)
        wone = ww[indone]
        print(f'\n Trace of generator eigenvalues close to unity according to tolerance {tol:.5f} --->  {sum(wone):.6f}')
        print(f' Factor of Phase Volume contraction {sum(abs(np.exp(wone)))/len(wone):.6f} per month\n')
        # Order retained unitary eigenvalues and obtain generator eigenvalues 'expwr'
        print(f'Number of Eigenvalues found {N1} over {len(ww)} for kmode tol {tol}\n')
        # _,expwr,indexpwr = order_w(wone,option='stable',direction='down',tdelta=1)

        headr = ('Mode','Eig-Real','Eig-Imag','Gen Real (Yr)','Gen Imag (Yr)')
        kaz_data = [(i,wone[i].real,wone[i].imag, 1./wone[i].real/12,   2*math.pi/wone[i].imag/12) for i in range(min(len(wone),50))]
        print(tab.tabulate(kaz_data, headr,stralign='center',tablefmt='github'))
        _vv = vv[:,indone]

        return wone,_vv,indone

    def compute_modes(self,ww,vv,modetype='',description=''):
        '''
        Compute Koopman Modes
        
        Parameters
        ----------
        
        ww:
            Koopman eigenvalues
        vv:
            Value of Koopman EIgenfunctions at samples
        modetype:
            Descriptive label 
        description:
            Data Descritpion
        
        
        Returns
        -------
        
        ds : data set
            .. table:: Content for `ds`
               :widths: auto

               ======================        ============================================
               Variable                      Description
               ======================        ============================================
               keig(time,modes)              Koopman eignfunction values at sample points
               kmodes(features,modes)        Koopman modes
               periods                       Periods of Koopman modes
               eigenvalues                   Koopman Eigenvalues
               ======================        ============================================
        '''
        
        Nmodes = vv.shape[1]
        Ntimes = self.ntime 
        Nfeatures = self.npoints
        # The ordering is consistent with Mezic
        Phiinv= sc.pinv(vv)
        KMM = Phiinv@(self.PsiX.T)
        print(f'Computed Kmodes {KMM.shape} for {vv.shape} eigenfunctions')
        # print(f'Computed KMM {Nmodes} {Nfeatures} vv  {Ntimes} {Nmodes} eigenfunctions')

        # Create Data Sets for Koopman modes
        ds = xr.Dataset(
            data_vars=dict(
            kmodes=(["x", "modes"], KMM.T),
            eigfun=(["time", "modes"], vv),
            periods=(["modes"],  2*math.pi/ww.imag/12),
            eigval=(["modes"],  ww )                                 ),

        coords=dict(
            feature=(["x"], np.arange(Nfeatures)),
            time=(["time"], np.arange(Ntimes)),
            modes=(["modes"], np.arange(Nmodes) ) ),
        attrs=dict(type=modetype, description=description)  )

        return ds

    def reconstruct_koopman(self,ds,decode=None,select_modes=None):
        '''
    
        Reconstruction of data from Koopman modes
        
        Parameters
        ----------
       
        ds:
            data set from `compute_modes`
        
        decode:
            Array decoding features
            
        decode: 
            If not empty contains the array for decoding
            to physical space
        
        select_modes: list
            If not empty reconstrut using only selected modes
            [2,7,4]
            

            
        Returns
        -------
        
        data : Array
            Reconstructed data
            
        '''

        if select_modes is None:
            y = ds.eigfun@ds.kmodes
        else:
            y = ds.eigfun[:,select_modes]@ds.kmodes[:,select_modes]

        #Check Reality
        if np.sum(y.imag**2) > 1.e-14:
             raise ValueError('reconstruct_koopman:  --- Array not Real ')
        else:
            y = y.real

        if decode is None:
            return y
        else:
            return decode@y.data.T
    def find_norm_modes(self,KM, KE, KEI, sort='sort',decode=None):
        '''
        Compute norms of eigenfunctions 
        eliminate complex conjugates, optionally sort in order
        of magnitude  

        Parameters
        ----------
        KM:
            Koopman Modes (KM)
        KE: 
            Koopman Eigenfunctions (KE)
        KEI:
            Koopman EIgenvalues (KEI)
        sort:
            'sort' -- get also sorted eigenvector in order of magnitude
        decode:
            None -- No decoding is applied, otherwise use array `decode`

        Returns
        -------

        (vnorm, KM, VM, KEI):
            Norms, Koopman Modes, Eigenfunctions, Eigenvalues
        (vnormsort, KM_sort, VM_sort, KEI_sort):
            Sorted Norms, Koopman Modes, Eigenfunctions, Eigenvalues
        '''

        # Eliminate complex conjugates
        ivar = np.where(KEI.imag >= 0)
        KM_single = KM[:,ivar[0]]
        VM_single = KE[:,ivar[0]]
        wti = KEI[ivar[0]]

        if decode is None:
            KM_out = KM_single.data
        else:
            KM_out =  decode@KM_single.data


        if sort == 'sort':
            vvar = np.diag(KM_single.T.conj().data@KM_single.data).real
            vnorm = vvar
            indnorm = np.argsort(vnorm)[::-1]
            return (vnorm,KM_out, VM_single, wti), (vnorm[indnorm],KM_out[:,indnorm], VM_single[:,indnorm], wti[indnorm])
        else:
            return (vnorm,KM_out, VM_single, wti)

class KEig():
    '''
    Class for computing  eigenfunctions values from coefficients
    '''
    __slots__ = ('k','X','n','m', 'v', 'nv','max','min','d')  


    def __init__(self, k, X, v, nv):
        '''
        Constructor for the eigenfunctionc objects
        '''
        self.k = k 
        '''Kernel '''
        self.X = X 
        '''Data matrix containing snapshots'''
        self.v = v
        '''Coefficients of eigenfunction'''
        self.nv = nv
        '''Order of the Eigenfunction'''
        self.n = X.shape[0]
        '''Dimension of state space'''
        self.m = X.shape[1]
        '''Number of snapshots'''
        self.max =  np.amax(v[:,nv] )
        '''Max of eigenfunction values'''
        self.min =  np.amin(v[:,nv] )
        '''Min of eigenfunction values''' 
        self.d =  self.max-self.min
        '''Distance of eigenfunction'''
        print(' Max {}  Min  {}'.format(self.max,self.min))

    def __call__(self, x):
        ''' Function evaluation.'''
        f = 0
        for i in range(self.m):
            f += self.v[i,self.nv]*self.k(x, self.X[:, i])
        return f

    def __repr__(self):
        '''  Printing Information '''
        print(' \n Eigenfunction for  {} \n'.format(self.k))
        print(' Number of Eigenfunction N={} \n'.format(self.nv))
        print(' Max={},  Min={} \n'.format(self.max,self.min))
        return  '\n'
        
    def fdfeval(self, x):
        '''Function and gradient evaluation.'''
        f = 0
        df = np.zeros((self.d,))
        for i in range(self.n):
            fi = self.v[i]*self.k(x, self.X[:, i])
            f += fi
            df -= 2/(2*self.k.sigma**2)*(x - self.X[:, i])*fi
        return f, df

    def xmax(self, fac):
        '''
        Extract States within a certain factor from Max
            Inputs:
                    fac     Distance from Max
            Return:
                    test    the average state
                    indv    the indices of the states averaged
        '''
               
        indb = self.v > self.max-self.d*fac
        
        indv =np.where(indb)[0]
        if len(indv) == 0:
            print("Error in xmax, no index found for value  {}".format(fac))
            return
        else:
            test=np.mean(self.X[:,indb],axis=1)           
            print('Indices of states for Max {}'.format(indv))
        return test,indv

    def xmin(self, fac):
        ''' 
        Extract States within a certain factor from Min
            Inputs:
                    fac     Distance from Max
            Return:
                    test    the average state
                    indv    the indices of the states averaged
        '''

        indb = self.v < self.min+self.d*fac
            
        indv =np.where(indb)[0]
        if len(indv) == 0:
            print("Error in xmin, no index found for value  {}".format(fac))
            return
        else:
            test=np.mean(self.X[:,indb],axis=1)          
            print('Indices of states for Min {}'.format(indv))
        return test,indv

    def xclose(self, val,tol=0.001):
        ''' 
        Extract States close a certain factor from max
            Return:
                    test    the average state
                    indv    the indices of the states averaged
        '''
        
        indb=abs(self.Gv - val) < tol 
        indv =np.where(indb)[0]
        if len(indv) == 0:
            print("Error in xmin, no index found close to value  {}".format(val))
            return
        else:
            test=np.mean(self.X[:,indb],axis=1)          
            print('Indices of states {} close to {}'.format(indv,val))
        return test,indv
        

