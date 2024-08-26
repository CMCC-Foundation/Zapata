'''
Auxiliary Plotting routines
===========================

Plotting routines for verification of the forecasts.

Utilities 
---------

'''
import os,sys
import math
import numpy as np
import numpy.linalg as lin  
import xarray as xr
import pandas as pd

import cartopy.crs as crs

import scipy.linalg as sc

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# import transformers as tr
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

import zapata.computation as zcom
import zapata.data as zd
import zapata.lib as zlib
import zapata.mapping as zmap
import zapata.koopman as zkop
import AIModels.AIClasses as zaic



def Single_Forecast_plots(  F, V,  field, level, cont, timestart, world_projection='Atlantic', maxtime=3, stride = 1,colorbars=True,\
                           title='', mainlabel='', arealatz=None, arealonz=None, centlon=0,labfile=None):
    '''
    Make verification plots for the field `field` and level `level` for the dataset `F` and `V`.
    Plots are made in two column of `maxtime` rows each.
    
    Parameters
    ==========
    
    F: xarray dataset
        Dataset with the forecast to be analyzed

    V: xarray dataset
        Dataset with the verification field

    mainlabel: string
        Main label for the plot

    field: string
        Field to be analyzed
    level: string
        Level to be analyzed
    timestart: string
        Starting date for the plotting
    maxtime: int
        Maximum number of time steps to be plotted
        The number of rows  will be determined by this number
    stride: int
        Stride for the plotting
    title: string
        Title for the plot
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    arealatz: numpy array
        Latitude limits of the field
    arealonz: numpy array
        Longitude limits of the field
    centlon: float
        Central longitude
    labfile: string
        Name of the file to be written
    
    Returns
    =======
    
    None
    
    '''
    
    label1 = mainlabel + field if level is None else mainlabel + field + level
    
    index = int(np.where(F['time'] == pd.Timestamp(timestart))[0]) 
    iv = index - 1 
    
    print(f'Plotting for time {F.time[iv].data} a total of {maxtime} time steps')   
    
    nrows = int(maxtime/stride)
    figy = 4*(nrows-3) + 8

    fig_index = np.arange(0,maxtime,stride,dtype=np.int32)
    fig,ax,pro=zmap.init_figure(nrows,2,world_projection, constrained_layout=False, figsize=(12,figy) )
    for i in range(len(fig_index)): 
        label12 = str(F.time[iv+fig_index[i]].data)[0:10]
        label13 = f'Forecast Lead {fig_index[i]}'
        out=zmap.xmap(F.isel(time=index,lead=fig_index[i]).unstack(),cont, pro, ax=ax[i,0],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), custom=True,
                    title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='seismic',contour=False)
        out.gl.right_labels = False
        if colorbars:
            zmap.add_colorbar(fig, out.handles['filled'], ax[i,0], label_size=10,edges=True)
        label13 = f'Verification Lead {fig_index[i]}'
        out2=zmap.xmap(V.isel(time=index,lead=fig_index[i]).unstack(),cont, pro, ax=ax[i,1],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), custom=True,
                    title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='seismic',contour=False)
        out2.gl.left_labels = False
        if colorbars:
            zmap.add_colorbar(fig, out2.handles['filled'], ax[i,1], label_size=10,edges=True)
    
    if not colorbars:
        # Create an axis on the bottom  of the figure for the colorbar
        
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])

        # Create a colorbar based on the second image
        cbar = fig.colorbar(out.handles['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')

    plt.suptitle(title,fontsize=20)
   
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape',  format='pdf')
    
    fig.subplots_adjust(wspace=0.05,hspace=0.0001)
    plt.show()
    return

def many_plots(  F,  field, level, cont, timestart, title='', mainlabel='', mode='time', lead=0, ncols=2, nrows=3, arealatz=None, arealonz=None, centlon=0,labfile=None):
    '''
    Make many plots for the field `field` and level `level` for the dataset `F`.
    Plots are made in rows and columns according to `ncol` and `nrows`.
    
    Parameters
    ==========
    
    F: xarray dataset
        Dataset with the field to be analyzed
    mainlabel: string
        Main label for the plot

    field: string
        Field to be analyzed
    level: string
        Level to be analyzed
    timestart: string
        Starting date for the plotting
    title: string
        Title for the plot
    lead: int
        Lead time for the plotting
    ncols: int
        Number of columns for the plotting
    nrows: int
        Number of rows for the plotting
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    arealatz: numpy array
        Latitude limits of the field
    arealonz: numpy array
        Longitude limits of the field
    centlon: float
        Central longitude
    labfile: string
        Name of the file to be written
    
    Returns
    =======
    
    None
    
    '''
    
    label1 = mainlabel + field + level
    

    index = int(np.where(F['time'] == pd.Timestamp(timestart))[0]) 
    iv = index - 1 + lead
    if mode == 'time':  
        print(f'Plotting for time {F.time[iv].data} and lead {lead}')
    elif mode == 'lead':
        lead = -1
        print(f'Plotting all leads for time {F.time[index].data}')
    else:
        print(f'No mode {mode} defined')
        return None

    fig,ax,pro=zmap.init_figure(nrows,ncols,'Atlantic', constrained_layout=False, figsize=(24,12) )
    for i in range(nrows):
        for j in range(ncols):
            if mode == 'time':  
                iv += 1
                label12 = str(F.time[iv].data)[0:10]
            elif mode == 'lead':
                lead += 1
                label12 = str(F.time[iv+lead].data)[0:10]
            else:
                return None
            
           
            label13 = f'Lead {lead}'
            handle=zmap.xmap(F.isel(time=iv,lead=lead).unstack(),cont, pro, ax=ax[i,j],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                    xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), 
                        title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='coolwarm',contour=False)
            zmap.add_colorbar(fig, handle['filled'], ax[i,j], label_size=10,edges=True)
    
    plt.suptitle(title,fontsize=20)
    # fig.subplots_adjust(wspace=0.1,hspace=0.1)
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape',  format='pdf')
    plt.show()
    return

def Forecast_plots(  F, V, D,  field, level, cont, timestart, leads=None,\
                   figsize=(12,8), world_projection='Atlantic',\
                   colorbars=True, title='', mainlabel='', picturelabels=None, arealatz=None, arealonz=None, centlon=0,labfile=None):
    '''
    Make verification plots for the field `field` and level `level` for the dataset `F` and `V`.
    Plots are made in two column of `maxtime` rows each.
    
    Parameters
    ==========
    
    F: xarray dataset
        Dataset with the forecast to be analyzed
    V: xarray dataset
        Dataset with the verification field 
    D: xarray dataset
        Dataset with the deterministic forecast
    figsize: tuple
        Size of the figure
    world_projection: string
        World Projection for the plot
    mainlabel: string
        Main label for the plot
    picturelabels:namedtuple
        List of labels for the pictures
    field: string
        Field to be analyzed
    level: string
        Level to be analyzed
    timestart: string
        Starting date for the plotting
    column: int
        Number of columns for the plotting
    leads: list(int)
        List of lead time to be plotted
    title: string
        Title for the plot
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    arealatz: numpy array
        Latitude limits of the field
    arealonz: numpy array
        Longitude limits of the field
    centlon: float
        Central longitude
    labfile: string
        Name of the file to be written
    
    Returns
    =======
    
    None
    
    '''

    column = len(leads)
    
    label1 = mainlabel + field if level is None else mainlabel + field + level
    
    index = int(np.where(F['time'] == pd.Timestamp(timestart))[0]) 
    iv = index - 1 
    print(f'Plotting for time {F.time[iv].data} a total of {len(leads)} lead times')
    
    _,dyntime,_ = D.shape
    print(f'dynamic time {dyntime}')
    
    fig,ax,pro=zmap.init_figure(3,column,world_projection, constrained_layout=False, figsize=figsize )
    for k, i in zip(range(len(leads)), leads):
        label12 = str(advance_date(V.time[iv].data[()],i))[0:10]
        label13 =  f'Forecast Lead {i}, {picturelabels[0][k]}' if picturelabels is not None else f'Forecast Lead {i}'
        out=zmap.xmap(F.isel(time=index,lead=i).unstack(),cont, pro, ax=ax[1,k],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), custom=True,
                    title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='seismic',contour=False)
        out = _fix_labels(out,k,column)
        if colorbars:
            zmap.add_colorbar(fig, out.handles['filled'], ax[i,k], label_size=10,edges=True)
        label13 = f'Verification Lead {i}'
        out2=zmap.xmap(V.isel(time=index,lead=i).unstack(),cont, pro, ax=ax[2,k],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), custom=True,
                    title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='seismic',contour=False)
        out2 = _fix_labels(out2,k,column)
        if colorbars:
            zmap.add_colorbar(fig, out2.handles['filled'], ax[2,k], label_size=10,edges=True)
        
        if i< dyntime:
            label13 = f'GCM Lead {i}, {picturelabels[1][k]}' if picturelabels is not None else f'GCM Lead {i}'
            out3=zmap.xmap(D.isel(time=index,lead=i).unstack(),cont, pro, ax=ax[0,k],refline=None, c_format='{:4.2f}',data_cent_lon=centlon,\
                    xlimit=(arealonz[0],arealonz[1]),ylimit=(arealatz[1],arealatz[0]), custom=True,
                        title={'maintitle':label1, 'lefttitle':label12,'righttitle':label13},cmap='seismic',contour=False)
            out3 = _fix_labels(out3,k,column)
            if colorbars:
                zmap.add_colorbar(fig, out3.handles['filled'], ax[0,k], label_size=10,edges=True)
        else:
            print(f'No GCM data for lead {i}')
            ax[0,k].remove()
    if not colorbars:
        # Create an axis on the bottom  of the figure for the colorbar 
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        # Create a colorbar based on the first image
        cbar = fig.colorbar(out.handles['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')

    plt.suptitle(title,fontsize=20)
   
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape',  format='pdf')
    
    fig.subplots_adjust(wspace=0.05,hspace=0.0001)
    plt.show()
    return

def advance_date(date,months):
    '''
    Advance a date by a number of months
    
    Parameters
    ==========
    
    date: string
        Date to be advanced
    months: int
        Number of months to be advanced
    
    Returns
    =======
    
    newdate: string
        Advanced date
    
    '''
    
    # Convert the 'ns' datetime64 object to a pandas.Timestamp
    date_obj_pd = pd.Timestamp(date)

    # Advance the date by one month using pandas DateOffset
    date_next_month = date_obj_pd + pd.DateOffset(months=months)

    # Convert the resulting date back to numpy.datetime64 with 'D' precision
    date_next_month_np = np.datetime64(date_next_month, 'D')
    return date_next_month_np

def _fix_labels(out,k,column):
    '''
    Fix the labels for the plot
    
    Parameters
    ==========
    
    out: object
        Object with the plot
    k: int
        Index for the column
    col: int
        Column for the plot
    
    Returns
    =======
    
    None
    
    '''
    out.gl.left_labels = False
    out.gl.right_labels = False
    if k == 0:
        out.gl.left_labels = True
    elif k == column-1:
        out.gl.right_labels = True
    
    return out 
