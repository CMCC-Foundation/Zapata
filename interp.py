'''
Interpolation for Ocean and Atmospheric Data 
============================================

Routines and methods for interpolation of ocean and atmospheric model
field, either on regular grids or rotated, multi-pole grids.
For the ocean, is set for the CMCC Ocean model at the nominal resolution of 0.25 
respectively.

The staggering requires different interpolators operators for scalar points at T-points 
and vector quantities carried at (u,v) points. For the moment a simple interpolation is carried out
but a more accurate vector interpolation is under development.

The interpolation is obtained by triangulation of the starting grid and seaprate interpolation to the new grid.
The weights are preserved and they can be used for repeated application of the same set of grids.

Classes 
-------
| **Atmosphere_Interpolator**
| **Ocean_Interpolator**



'''

import os
import math
import numpy as np
import pickle
import gzip

import scipy.linalg as sc
import scipy.special as sp
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.spatial import Delaunay


import zapata.lib as zlib

import xarray as xr


class Atmosphere_Interpolator():
    """ 
    This class creates weights for interpolation of atmospheric fields.
    No mask is used.

    This class create an interpolator operator to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Parameters
    ----------

    grid : str
        Choice of output grids  

        * `1x1`  -- Regular 1 degree 
        * `025x025`  -- Coupled model grid, nominally 0.25, 

    option : str
        'linear', interpolation method

    Attributes
    ----------

    name : str
        Name of the interpolator
    grid : str
        Option for the grid
    option : str
        Interpolation method

    
    Notes
    =====

    It is a thin wrapper around `xarray` `interp_like` method.
    
    Examples    
    --------    
    Create the weights for interpolation

    >>> w= zint.Atmosphere_Interpolator('1x1','linear') 
    
    Interpolate temperature

    >>> target_xarray=w.interp_f(src_xarray)

    
    """

    __slots__ = ('name','tgt','choice')

    

    def __init__(self, grid, option='linear'):
        self.choice = option
        ''' str: Interpolation method selected `linear` or `nearest`'''
        # Put here info on grids to be obtained from __call__
        self.name = 'Atmosphere_Interpolator'
        '''str: Name of the Interpolator'''
        if grid == '1x1':
            # Selected regular 1 Degree grid
            lon1x1 = np.linspace(0,359,360)
            lat1x1 = np.linspace(-90,90,180)
            mm=np.ones([lat1x1.shape[0],lon1x1.shape[0]])
            tgt = xr.DataArray(mm,dims=['lat','lon'],\
                            coords={'lat':lat1x1,'lon':lon1x1})  
        elif grid == '025x025':
            homedir = os.path.expanduser("~")
            file = homedir + '/Dropbox (CMCC)/data_zapata/'+ 'masks_CMCC-CM2_VHR4_AGCM.nc'
            dst=xr.open_dataset(file,decode_times=False)
            lat25 = dst.yc.data[:,0]
            lon25 = dst.xc.data[0,:]
            tgt = xr.DataArray(dst.mask,dims=['lat','lon'],\
                            coords={'lat':lat25,'lon':lon25})  
        else:
            SystemError(f'Wrong Option in {self.name} --> {grid}')  
        self.tgt = tgt
        '''xarray: Target grid'''    
        return
        
    def __call__(self):
        return

    def __repr__(self):
        '''  Printing other info '''
        return '\n' 
    
    def interp_scalar(self, xdata):
        ''' 
        Perform interpolation  to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        
        '''
        
        res = xdata.interp_like(self.tgt,method=self.choice)
        return res

class Ocean_Interpolator():
    """This class creates weights for interpolation of ocean fields.

    This class create an interpolator operator from a *source grid* 
    `src_grid` to a *target grid* `tgt_grd`. The interpolator then
    can be used to perform the actual interpolation. The model uses
    an Arakawa C-grid, that is shown in the following picture. The f-points
    correspond to the points where the Coriolis terms are carried.

    .. image:: ../resources/NEMOgrid.png
        :scale: 25 %
        :align: right

    The Arakawa C-grid used in the ocean model show also the ordering
    of the points, indicating which points correspond to the (i,j) index.

    The *source grid* must be a `xarray` `DataSet` containing coordinates
    `latitude` and `longitude`.

    The *target grid* must be a `xarray` `DataArray` with variables `lat` and `lon`

    Border land points at all levels are covered by a convolution value using a
    window that can be changed in `sea_over_land`.
   
    Works only on single `DataArray`
    
    Parameters
    ----------

    src_grid : xarray
        Source grid
    tgt_grid : xarray
        Target grid
    level : float
        Depth to generate the interpolator
    window : int
        Window for sea over land
    period : int
        Minimum number of points in the sea-over-land process
    verbose: bool
        Lots of output

    Attributes
    ----------

    tgt_grid :
        Target grid 
    mask :
        Mask of the target grid
    vt :
        Weights
    wt :
        Weights
    mdir :
        Directory for masks files
    ingrid :
        Input Grid, `src_grid_name`
    outgrid :
        Output Grid `tgt_grid_name`
    window : 
        Window for convolution Sea-Over-Land (default 3)
    period :
        Minimum number of points into the window (default 1)
    T_lon :
        Longitudes of input T-mask
    T_lat : 
        Latitudes of input T-mask
    U_lon :
        Longitudes of input U-mask
    U_lat : 
        Latitudes of input U-mask
    V_lon :
        Longitudes of input V-mask
    V_lat : 
        Latitudes of input V-mask
    tangle :
        Angles of the T points of the input grid
    mask_reg :
        Mask of the Target grid
    cent_long :
        Central Longitude of the Target grid
    name :
        Name of the Interpolator Object
    level :
        Level of the Interpolator Object


    Methods
    -------

    Interp_T :
        Interpolate Scalar quantities at T points
    
    Interp_UV :
        Interpolate Vector Velocities at (U,V)

    mask_sea_over_land :
        Mask border point for Sea over land
    
    UV_sea_over_land :
        Fill U,V values over land
    
    to_file :
        Writes interpolator object to file (pickled format)
    
    Examples    
    --------    
    Create the weights for interpolation

    >>> w= zint.Ocean_Interpolator(src_grid,tgt_grid) 
    
    Interpolate temperature

    >>> target_xarray=w.interp_T(src_xarray,method='linear')

    Interpolate U,V

    >>> target_xarray=w.interp_UV(U_xarray,V_xarray,method='linear')


    """

    __slots__ = ('mask','masku','maskv','mdir', 'mask_reg', \
                'sea_index','sea_index_U','sea_index_V', \
                'latlon', 'masT_vec', 'maskub','maskvb','masktb',\
                'latlon_reg','sea_index_reg','regmask_vec', \
                'T_lat','T_lon','U_lat','U_lon','V_lat','V_lon',\
                'name','cent_long','tri_sea_T','tri_sea_U','tri_sea_V','tangle',\
                    'ingrid','outgrid','level','window','period')

    def __init__(self, src_grid_name, tgt_grid_name,level=1,verbose=False,window=3,period=1):
        # Put here info on grids to be obtained from __call__
        # This currently works with mask files
        # 'masks_CMCC-CM2_VHR4_AGCM.nc'
        # and
        # 'ORCA025L50_mesh_mask.nc'
        
        # Path to auxiliary Ocean files
        homedir = os.path.expanduser("~")
        self.mdir = homedir + '/Dropbox (CMCC)/data_zapata'
        self.ingrid = src_grid_name
        self.outgrid = tgt_grid_name
        
        # Parameter for sea over land
        self.window = window
        self.period = period
     
        # Check levels
        if level > 0:
            self.level=level
        else:
            SystemError(f' Surface variable not available, Level {level}')

        #Resolve grids
        s_in = self._resolve_grid(src_grid_name,level)

        tk = s_in['tmask']
        self.T_lon = s_in['lonT']
        self.T_lat = s_in['latT']
        mask = tk.assign_coords({'lat':self.T_lat,'lon':self.T_lon}).drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})


        tk = s_in['umask']
        self.U_lon = s_in['lonU']
        self.U_lat = s_in['latU']
        masku = tk.assign_coords({'lat':self.U_lat,'lon':self.U_lon}).drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        
        tk = s_in['vmask']
        self.V_lon = s_in['lonV']
        self.V_lat = s_in['latV']
        maskv = tk.assign_coords({'lat':self.V_lat,'lon':self.V_lon}).drop_vars(['U_lon','U_lat','V_lon','V_lat','T_lon','T_lat']).rename({'z':'deptht'})

        self.name = 'UV  Velocity'
        self.level = str(level)
      
        #Fix Polar Fold
        mask[-3:,:] = False

        #T angles
        self.tangle = s_in['tangle']
      
        #Sea over land
       
        self.mask = xr.where(mask !=0, 1,np.nan)
        self.masktb,dum = self.mask_sea_over_land(mask)
    
        self.maskub,self.masku = self.mask_sea_over_land(masku)
        self.maskvb,self.maskv = self.mask_sea_over_land(maskv)
 
        print(f' Generating interpolator for {self.name}')
        if verbose:
            print(self.mask,self.masku,self.maskv)
        
        # Get triangulation for all grids
        self.latlon,self.sea_index, self.masT_vec = get_sea(self.mask)
        self.tri_sea_T  = Delaunay(self.latlon)  # Compute the triangulation for T
        print(f' computing the triangulation for T grid')

        latlon_U,self.sea_index_U, masU_vec = get_sea(self.masku)
        self.tri_sea_U  = Delaunay(latlon_U)  # Compute the triangulation for U
        print(f' computing the triangulation for U grid')

        latlon_V,self.sea_index_V, masT_vec = get_sea(self.maskv)
        self.tri_sea_V  = Delaunay(latlon_V)  # Compute the triangulation for V
        print(f' computing the triangulation for V grid')
        
        #Target Grid

        s_out = self._resolve_grid(tgt_grid_name,level,verbose=verbose)
        self.mask_reg = s_out['tmask']
        self.cent_long = s_out['cent_long']
        self.latlon_reg,self.sea_index_reg,self.regmask_vec = get_sea(self.mask_reg)
        

    def __call__(self):
        print(f' Interpolator for T,U,V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        print(f' Main methods interp_T and interp_UV')
        return

    def __repr__(self):
        '''  Printing other info '''
        print(f' Interpolator for T,U,V GLORS data ')
        print(f' This is for level at depth {self.level} m')
        return '\n' 
    
    def interp_T(self, xdata, method='linear'):
        '''
        
        Perform interpolation for T Grid point to the target grid.
        This methods can be used for scalar quantities.

        Parameters
        ----------
        xdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        method : str    
            Method for interpolation    
                * 'linear'  , Use linear interpolation
                * 'nearest' , use nearest interpolation

        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        '''

        # Compute interpolation T
        Tstack = xdata.stack(ind=('y','x'))
        sea_T = Tstack[self.sea_index]
        temp = xr.full_like(self.regmask_vec,np.nan)
        if method == 'linear':
            interpolator = spint.LinearNDInterpolator(self.tri_sea_T, sea_T)
        elif method == 'nearest':
            interpolator = spint.NearestNDInterpolator(self.tri_sea_T, sea_T)
        else:
            SystemError(f' Error in interp_T , wrong method  {method}')
        T_reg = interpolator(self.latlon_reg)
        temp[self.sea_index_reg] = T_reg.data
        out = temp.unstack()
        #Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            out[:,1439] = out[:,1438] + delx*(out[:,1]-out[:,1438])/ddelx
            out[:,0] = out[:,1438] + 2*delx*(out[:,1]-out[:,1438])/ddelx
        return out

    def interp_UV(self, udata, vdata, method = 'linear'):
        '''
        Perform interpolation for U,V Grid point to the target grid.
        This methods can be used for vector quantities.

        The present method interpolates the U,V points to the T points, 
        rotates them and then interpolates to the target grid.

        Parameters
        ----------
        udata,vdata :  xarray
            2D array to be interpolated, it must be on the `src_grid`
        
        Returns
        -------
        out :  xarray
            Interpolated xarray on the target grid
        '''
        
        # Insert NaN
        udata = xr.where(udata < 200,udata, np.nan)
        vdata = xr.where(vdata < 200,vdata, np.nan)

        #udata,vdata = self.UV_sea_over_land(udata, vdata, self.masku,self.maskv)
        udata = self.fill_sea_over_land(udata,self.masku)
        vdata = self.fill_sea_over_land(vdata,self.maskv)

        # Compute interpolation  for U,V
        Ustack = udata.stack(ind=('y','x'))
        Vstack = vdata.stack(ind=('y','x'))
        sea_U = Ustack[self.sea_index_U]
        sea_V = Vstack[self.sea_index_V]

        #Interpolate to T_grid
        if method == 'linear':
            int_U = spint.LinearNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.LinearNDInterpolator(self.tri_sea_V, sea_V)
        elif method == 'nearest':
            int_U = spint.NearestNDInterpolator(self.tri_sea_U, sea_U)
            int_V = spint.NearestNDInterpolator(self.tri_sea_V, sea_V)
        else:
            SystemError(f' Error in interp_UV , wrong method  {method}')
        
        U_on_T = int_U(self.latlon)
        V_on_T = int_V(self.latlon)
        
        UT = self.masT_vec.copy()
        VT = self.masT_vec.copy()
        UT[self.sea_index] = U_on_T.data
        VT[self.sea_index] = V_on_T.data
        
        UT = UT.unstack()
        VT = VT.unstack()

        #Rotate Velocities
        fac = self.tangle*math.pi/180.
        uu = (UT * np.cos(fac) - VT * np.sin(fac)).drop_vars(['nav_lon','nav_lat'])
        vv = (VT * np.cos(fac) + UT * np.sin(fac)).drop_vars(['nav_lon','nav_lat'])

        # Interpolate to regular grid
        Uf = self.interp_T( uu, method=method)
        Vf = self.interp_T( vv, method=method)

        #Fix dateline problem
        if self.outgrid == 'L44_025_REG_GLO':
            delx=0.25
            ddelx=3*delx
            Uf[:,1439] = Uf[:,1438] + delx*(Uf[:,1]-Uf[:,1438])/ddelx
            Uf[:,0] = Uf[:,1438] + 2*delx*(Uf[:,1]-Uf[:,1438])/ddelx
           
        
        return Uf,Vf
    
    def to_file(self, filename):
        '''
        This method writes to file the interpolating object in
        pickled format
        '''

        with gzip.open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def mask_sea_over_land(self,mask):
        '''
        Mask border point for `Sea over land`.

        Sea over land is obtained by forward and backward filling NaN land
        value with an arbitrary value, then athey are masked to reveal only the 
        coastal points.

        Parameters
        ==========
        mask:
            original mask
        
        Returns
        =======
        border:
            border mask

        '''
        masknan=xr.where(mask != 0, 1, np.nan)
        um1 = masknan.ffill(dim='x',limit=1).fillna(0.)+  masknan.bfill(dim='x',limit=1).fillna(0.)- 2*masknan.fillna(0)
        um2 = masknan.ffill(dim='y',limit=1).fillna(0.)+  masknan.bfill(dim='y',limit=1).fillna(0.)- 2*masknan.fillna(0)
        bord = (um1+um2)/2
        um = bord + masknan.fillna(0)
        um = xr.where(um!=0, 1,np.nan)
        mb = xr.where(bord !=0, 1,np.nan)

        return mb, um

    def _resolve_grid(self,ingrid,level,verbose=False):
        '''
        Internal routine to resolve grid informations
        '''
        
        if ingrid == 'L75_025_TRP_GLO':
            print(f' Tripolar L75 0.25 Grid -- {ingrid}')  
            grid = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/tmask_UVT_latlon_coordinates.nc').sel(z=level)
            geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            angle = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/ORCA025L75_angle.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': geo.glamt,'latT':geo.gphit,'lonU':geo.glamu, \
                    'latU':geo.gphiu,'lonV':geo.glamv,'latV':geo.gphiv  }
        
        elif ingrid == 'L44_025_TRP_GLO':
            print(f' Tripolar L44 0.25 Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L44_025_TRP_GLO/tmask44_UVT_latlon_coordinates.nc').sel(z=level)
            angle = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/ORCA025L75_angle.nc')
            geo = xr.open_dataset(self.mdir + '/L75_025_TRP_GLO/NEMO_coordinates.nc')
            struct={'tmask': grid.tmask, 'umask': grid.umask,'vmask': grid.vmask, 'tangle': angle.tangle, \
                    'lonT': geo.glamt,'latT':geo.gphit,'lonU':geo.glamu, \
                    'latU':geo.gphiu,'lonV':geo.glamv,'latV':geo.gphiv  }
    
        elif ingrid == 'L75_025_REG_GLO':
            print(f' Regular L75 0.25 Lat-Lon Grid -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/L75_025_REG_GLO/GLO-MFC_001_025_mask_bathy.nc').sel(z=level). \
                rename({'longitude':'T_lon','latitude':'T_lat'})
            struct={'tmask': grid.mask, 'tangle': None }
        
        elif ingrid == 'L44_025_REG_GLO':
            print(f' Regular L44 0.25 Lat-Lon Grid from WOA -- {ingrid}')
            grid = xr.open_dataset(self.mdir + '/WOA/m025x025L44.nc').sel(depth=level)
            cent_long = 720
            struct={'tmask': grid.m025x025L44, 'tangle': None ,'cent_long' : cent_long}
        
        else:
             SystemError(f'Wrong Option in _resolve_grid --> {ingrid}')  
        
        if verbose:
            print(f'Elements of {ingrid} extracted \n ')
            for i in struct.keys():
              print(f' {i} \n')
        return struct
    def fill_sea_over_land(self,U,u_mask):
        '''
        Put values Sea over land.

        Using the mask of the border points, the border points are filled 
        with the convolution in 2D, using a window of width `window`, here
        The min_periods value is controlling the minimum number of points
        within the window that is necessary to yield a result.

        They can be fixed as attributes of the interpolator

        Parameters
        ==========

        U:
            Field to be treated
        u_mask:
            Mask for the field

        Returns
        =======
        U:
            Filled array
        '''
        
        r = U.rolling(x=self.window, y=self.window, min_periods=self.period,center=True)
        r1=r.mean()
        
        border = ~np.isnan(self.maskub).drop_vars({'lat','lon'}).stack(ind=self.maskub.dims)
        UU=U.stack(ind=U.dims)
        rs = r1.stack(ind=r1.dims)
        UU[border] = rs[border]

        UUU = UU.unstack()
        UUU = UUU.assign_coords({'lon':u_mask.lon,'lat':u_mask.lat})

        return UUU
    

def get_sea(maskT):
    '''
    Obtain indexed coordinates for sea points
    '''

    # Try Interpolation
    tm = zlib.putna(-0.1,0.1,maskT)
    sea_index = ~np.isnan(tm).stack(ind=maskT.dims)

    maskT_vec = tm.stack(ind=maskT.dims)
    land_point = maskT_vec[~sea_index]
    sea_point = maskT_vec[sea_index]
    land_point.name = 'Land'
    sea_point.name = 'Sea'
    
    # Compute triangularization for sea points
    latlon=np.asarray([sea_point.lat.data,sea_point.lon.data]).T
    return latlon, sea_index, maskT_vec
def from_file(file):
    '''
    Read interpolator object from `file`

    '''
    with gzip.open(file, 'rb') as input:
        w = pickle.load(input)
    return w

    