'''
Auxiliary Plotting routines
===========================

Plotting routines for verification of the forecasts.

Functions   


'''
import os,sys
import math
import numpy as np
import numpy.linalg as lin  
import xarray as xr
import pandas as pd

import cartopy.crs as crs

import scipy.linalg as sc
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import csv
import pickle


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

import AIModels.AIutil as zai



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
    
    if maxtime > stride:
        nrows = int(maxtime/stride)
    else:
        nrows = 1
        stride = maxtime

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

def many_plots(F, field, level, cont, timestart, world_projection='Pacific', mainlabel='', mode='time', lead=0, ncols=2, nrows=3, arealatz=None, arealonz=None, centlon=0, labfile=None, colorbars=False, suptitle=''):
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
    mode: string
        Mode for the plotting, either 'time' or 'lead'
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
    colorbars: bool
        Whether to add colorbars to each panel or a single colorbar at the bottom
    suptitle: string
        General title on top of the plot
    
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

    fig, ax, pro = zmap.init_figure(nrows, ncols, world_projection, constrained_layout=False, figsize=(24, 12))
    for i in range(nrows):
        for j in range(ncols):
            if mode == 'time':  
                iv += 1
                label12 = str(F.time[iv].data)[0:10]
            elif mode == 'lead':
                lead += 1
                label12 = str(F.time[iv + lead].data)[0:10]
            else:
                return None
                        
            if 'lead' in F.dims:
                label13 = f'Lead {lead}'
                handle = zmap.xmap(F.isel(time=iv, lead=lead).unstack(), cont, pro, ax=ax[i, j], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                                   xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                                   title={'maintitle': label1, 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            else:
                label13 = f'Time {label12}'
                handle = zmap.xmap(F.isel(time=iv).unstack(), cont, pro, ax=ax[i, j], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                                   xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                                   title={'maintitle': label1, 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            if colorbars:
                zmap.add_colorbar(fig, handle['filled'], ax[i, j], label_size=10, edges=True)
    
    if not colorbars:
        # Create an axis on the bottom of the figure for the colorbar
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        # Create a colorbar based on the first image
        cbar = fig.colorbar(handle['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')

    plt.suptitle(suptitle, fontsize=20)
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape', format='pdf')
    plt.show()
    return

def _fix_labels(out,side='right'):
    '''
    Fix the labels for the plot
    
    Parameters
    ==========
    
    out: object
        Object with the plot
    side: string
        Side for the labels
    
    Returns
    =======
    
    None
    
    '''
    out.gl.left_labels = False
    out.gl.right_labels = False
    match side:
        case 'left':
            out.gl.left_labels = True
        case 'right':
            out.gl.right_labels = True
    
    return out 

def Forecast_plots(F, V, field, level, cont, world_projection='Atlantic', pictimes=None, colorbars=True,
                   title='', mainlabel='', arealatz=None, arealonz=None, centlon=0, labfile=None):
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
    maxtime: int
        Maximum number of time steps to be plotted
        The number of rows will be determined by this number
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
   
    
    # if maxtime > stride:
    #     nrows = int(maxtime / stride)
    # else:
    #     nrows = 1
    #     stride = maxtime
    nrows = len(pictimes)
    figy = 4 * (nrows - 3) + 8

    fig_index = pictimes
    fig, ax, pro = zmap.init_figure(nrows, 2, world_projection, constrained_layout=False, figsize=(12, figy))
    for i in range(len(fig_index)): 
        label12 = str(F.time[fig_index[i]].data)[0:10]
        label13 = f'Forecast Lead {fig_index[i]}'
        out = zmap.xmap(F.isel(time=fig_index[i]).unstack(), cont, pro, ax=ax[i, 0], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                        xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), custom=True,
                        title={'maintitle': label1, 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
        out.gl.right_labels = False
        if colorbars:
            zmap.add_colorbar(fig, out.handles['filled'], ax[i, 0], label_size=10, edges=True)
        label12 = str(V.time[fig_index[i]].data)[0:10]
        label13 = f'Verification Lead {fig_index[i]}'
        out2 = zmap.xmap(V.isel(time=fig_index[i]).unstack(), cont, pro, ax=ax[i, 1], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                         xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), custom=True,
                         title={'maintitle': label1, 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
        out2.gl.left_labels = False
        if colorbars:
            zmap.add_colorbar(fig, out2.handles['filled'], ax[i, 1], label_size=10, edges=True)
    
    if not colorbars:
        # Create an axis on the bottom of the figure for the colorbar
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])

        # Create a colorbar based on the second image
        cbar = fig.colorbar(out.handles['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')

    plt.suptitle(title, fontsize=20)
   
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape', format='pdf')
    
    fig.subplots_adjust(wspace=0.05, hspace=0.0001)
    plt.show()
    return

def Three_Forecast_plots(  F, V, D,  field, level, cont, timestart, pictimes,\
                   figsize=(12,8), world_projection='Atlantic',\
                   colorbars=True, maintitle='', picturelabels=None, arealatz=None, arealonz=None, centlon=0,labfile=None):
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
    maintitle: string
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
    pictimes: list
        List of times to be plotted
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

    column = len(pictimes)
    maxV = len(V['time'])
    maxD = len(D['time'])
    maxF = len(F['time'])

    title = maintitle + field if level is None else maintitle + field + level
   
    
    index = int(np.where(F['time'] == pd.Timestamp(timestart))[0]) 
    iv = index 
    
    fstyle = {'fontsize': 11, 'fontfamily': 'futura', 'fontweight': 'bold'}
    tstyle = {'fontsize': 12, 'fontfamily': 'futura', 'fontweight': 'bold'}
    sup_style = {'fontsize': 24, 'fontfamily': 'futura', 'fontweight': 'bold'}
   
    
    fig,ax,pro=zmap.init_figure(3,column,world_projection, constrained_layout=False, figsize=figsize )
    for k, i in enumerate(pictimes):
        if i <= maxF:
            label12 = str(F.time[i].data)[0:10]
            label13 = f'{picturelabels[0][k]}' if picturelabels is not None else f'Lead Month {i}'
            out = zmap.xmap(F.isel(time=i).unstack(), cont, pro, ax=ax[k, 0], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                            xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                            custom=True, label_style=fstyle, title_style=tstyle,
                            title={'maintitle': 'Forecast', 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            out = _fix_labels(out, 'left')
            if colorbars:
                zmap.add_colorbar(fig, out.handles['filled'], ax[k, 0], label_size=10, edges=True)
        else:
            ax[k, 0].axis('off')
        
        if i <= maxD:
            label13 = f'{picturelabels[1][k]}' if picturelabels is not None else f'Lead Month {i}'
            out2 = zmap.xmap(D.isel(time=i).unstack(), cont, pro, ax=ax[k, 1], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                            xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                            custom=True, label_style=fstyle, title_style=tstyle,
                            title={'maintitle': 'GCM', 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            out2 = _fix_labels(out2, None)
            if colorbars:
                zmap.add_colorbar(fig, out2.handles['filled'], ax[k, 1], label_size=10, edges=True)
        else:
            ax[k, 1].axis('off')

        if i <= maxV:
            label13 = f'{picturelabels[2][k]}' if picturelabels is not None else ''
            out3 = zmap.xmap(V.isel(time=i).unstack(), cont, pro, ax=ax[k, 2], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                            xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                            custom=True, label_style=fstyle, title_style=tstyle,
                            title={'maintitle': 'Obs', 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            out3 = _fix_labels(out3, 'right')
            if colorbars:
                zmap.add_colorbar(fig, out3.handles['filled'], ax[k, 2], label_size=10, edges=True)
        else:
            ax[k, 2].axis('off')

        # Fix properties of bounding box
        zmap.changebox(out.ax,'all',linewidth=2,color='black',capstyle='round')
        zmap.changebox(out2.ax,'all',linewidth=2,color='black',capstyle='round')
        zmap.changebox(out3.ax,'all',linewidth=2,color='black',capstyle='round')

    if not colorbars:
        # Create an axis on the bottom  of the figure for the colorbar 
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        # Create a colorbar based on the first image
        cbar = fig.colorbar(out.handles['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')
    
        
        
    plt.suptitle(title,**sup_style)
   
    if labfile is not None:
        plt.savefig(labfile, orientation='landscape',  format='pdf')
    
    
    # fig.subplots_adjust(wspace=0.05,hspace=0.0001)
    plt.show()
    return ax
# Write a routine to selecct indeces corrsponding to  ceartain months into a xarray of datatimes64
def select_months(datetimes,months):
    '''
    Select the indices corresponding to certain months in a xarray of datetimes64
    
    Parameters
    ==========
    
    datetimes: xarray of datetimes64
        Datetimes to be analyzed
    months: string or list of months
        Label of months to be selected
    
    Returns
    =======
    
    indices: numpy array
        Indices of the selected months
    
    '''
    if isinstance(months, str):
        match months:
            case 'JFM':
                months = [1,2,3 ]
            case 'AMJ':
                months = [4,5,6 ]
            case 'JAS':
                months = [7,8,9 ]
            case 'OND':
                months = [10,11,12 ]
            case 'DJF':
                months = [12,1,2 ]
    else:
        months = list(months)
    
    indices = np.array([],dtype=np.int32)
    for month in months:
        indices = np.concatenate((indices,np.where(datetimes.dt.month == month)[0]))
    return indices

def plot_skill(corrresult,persistence,rmsres,rmsper,rmsdyn=None,corrdyn=None,labtit=None,
               batch=False, savefig=False, data_dict=None,
               skill='mean',numbers=True, printout=False, labfile='scores',figsize=(10,10)): 
    '''
    Makes plot of the skill scores
    for forecasts and persistence
    Optionally add the skill scores for the dynamic forecast
    '''
    
    Tpredict = corrresult.shape[0]-1
    tim = np.arange(0,Tpredict+1)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, width_ratios=[0.5], height_ratios=2*[1], wspace=0.3, hspace=0.3)

    #  panels
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    #  plot the skill scores and choose type of skill for correlation
    match skill:
        case 'mean':
            sk = np.mean(corrresult,axis=1)
            ps = np.mean(persistence,axis=1)
            if corrdyn is not None:
                dyn = np.mean(corrdyn,axis=1)  # Start from month 1
        case 'median':
            sk = np.median(corrresult,axis=1)
            ps = np.median(persistence,axis=1)
            if corrdyn is not None:
                dyn = np.median(corrdyn,axis=1)  # Start from month 1
        case _:
            raise ValueError('Invalid skill choice for acc')
    if printout:
        write_skill_to_csv(labfile, skill, Tpredict, sk, ps, corrdyn=dyn, data_dict=data_dict)
    
    if not batch:
        ax1.plot(tim,sk)
        ax1.plot(tim,ps,linestyle='dashed')
        if corrdyn is not None:
            ax1.plot(tim[0:7],dyn,linestyle='dashed',color='green')
            
        ax1.axhline(0.6,linestyle='dashed')
        ax1.set_title(f'ACC score {skill}',loc='left')
        ax1.set_title(labtit,loc='right')

        if numbers:
            for ii in range(0,Tpredict+1):
                ax1.text(ii,min(ax1.get_ylim())+0.1,f'{sk[ii]:4.2f}',horizontalalignment='center')
                ax1.text(ii,min(ax1.get_ylim())+0.2,f'{ps[ii]:4.2f}',horizontalalignment='center',color='coral')
        
    #  plot the skill scores and choose type of skill for correlation
    match skill:
        case 'mean':
            sk = np.mean(rmsres,axis=1)
            ps = np.mean(rmsper,axis=1)
            if rmsdyn is not None:
                dyn = np.mean(rmsdyn,axis=1)  # Start from month 1
        case 'median':
            sk = np.median(rmsres,axis=1)
            ps = np.median(rmsper,axis=1)
            if rmsdyn is not None:
                dyn = np.median(rmsdyn,axis=1)  # Start from month 1
        case _:
            raise ValueError('Invalid skill choice for rms')
    
    
    if printout:
        labrms = labfile.replace("ACC", "RMS")
        write_skill_to_csv(labrms, skill, Tpredict, sk, ps, corrdyn=dyn, data_dict=data_dict)
   

    if not batch:
        ax2.plot(tim,sk)
        ax2.plot(tim,ps,linestyle='dashed')
        if corrdyn is not None:
            ax2.plot(tim[0:7],dyn,linestyle='dashed',color='green')
            
        ax2.set_title(f'RMS score {skill}',loc='left')
        ax2.set_title(labtit,loc='right')
        if numbers:
            for ii in range(0,Tpredict+1):
                ax2.text(ii,min(ax2.get_ylim())+0.01,f'{sk[ii]:4.2f}',horizontalalignment='center')
                ax2.text(ii,min(ax2.get_ylim())+0.02,f'{ps[ii]:4.2f}',horizontalalignment='center',color='coral')
            
        if savefig:
            plt.savefig(f'{labfile}.pdf', format='pdf')
        plt.show()
    return

def extract_and_merge_csv(input_folder, output_file, selected_rows):
    """
    Extract selected rows from multiple CSV files in a folder and merge them into a single CSV.
    
    Parameters
    ----------
    input_folder : str
        Path to the folder containing CSV files.
    output_file : str
        Path to the output merged CSV file.
    selected_rows : list of int
        List of row indices to extract from each CSV.
    
    Returns
    -------
    None
        Saves the merged CSV file to the specified output path.
    """
    merged_df = pd.DataFrame()
    # listfile = ['ACC_Score_Y620d_7_99_mean.csv','ACC_Score_Y620d_7_99_mean.csv','ACC_Score_Y620d_7_99_mean.csv']
    header_saved = False
    column_names = None

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            if not header_saved:
                df = pd.read_csv(file_path, header=0)
                column_names = df.columns
                header_saved = True
            else:
                df = pd.read_csv(file_path, header=None, skiprows=1, names=column_names)

            selected_df = df.iloc[selected_rows]
            merged_df = pd.concat([merged_df, selected_df], ignore_index=True)

    
    # Save the merged data to a CSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        merged_df.to_csv(f)#, index=False, header=True)
    
    print(f"Merged CSV saved as {output_file}")

   
def write_skill_to_csv(labfile, skill, Tpredict, sk, ps, corrdyn=None, data_dict=None):
    """
    Write skill scores to a CSV file.

    Parameters
    ----------
    labfile : str
        Base name for the output CSV file.
    skill : str
        Skill type (e.g., 'mean', 'median').
    Tpredict : int
        Number of prediction time steps.
    sk : array-like
        Skill scores for the forecast.
    ps : array-like
        Skill scores for the persistence.
    corrdyn : array-like, optional
        Skill scores for the dynamic forecast, by default None.

    Returns
    -------
    None
    """
    vars = zai.transform_strings(data_dict['input_fields'])
    csv_filename = f"{labfile}_{skill}.csv"
    print(f"Writing data to {csv_filename}")
    head = ['Subcase_var','Subcase','VARS','LAGS','HID','LAYERS','EOF','DISCOUNT','INSEQ','AREA']
    expvalue = [str(data_dict['subcase_var']), str(data_dict['subcase']), str(vars), str(data_dict['params']['lags']), str(data_dict['params']['D_DIM']), str(data_dict['params']['enc_dec_layers']), str(data_dict['eof']), str(data_dict['params']['discount']), str(data_dict['params']['TIN']), str(data_dict['area'])]
    print(expvalue)
    print(head)
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'Exp'] + head + [f'M{ii}' for ii in range(0, Tpredict + 1)])
        writer.writerow([f'AC-{labfile}'] + expvalue +[f"{val:.4f}" for val in sk])
        writer.writerow([f'PE-{labfile}'] + expvalue + [f"{val:.4f}" for val in ps])
        if corrdyn is not None:
            writer.writerow([f'DY-{labfile}'] + len(expvalue)*[' ']+ [f"{val:.2f}" for val in corrdyn])

    return

def plot_csv(file, figsize=(10, 6), colname='Exp', title='', xlab=None, ylab=None,
             row_indices=[0, 3, 6, 9, 12], per_indices=None, GCM_indices=None,
             sort_rows=False, savefile=None, input_axes=None, line06=True,
             lstyle=None,tstyle=None):
    '''
    Plot the data in the skill csv file
    
    Parameters
    ==========
    file: str
        Path to the CSV file
    figsize: tuple
        Size of the figure
    colname: str
        Column name to be used as labels
    title: str
        General title for the plot
    xlab: str
        Label for the x-axis
    ylab: str
        Label for the y-axis
    row_indices: list
        Indices of the rows to be plotted
    per_indices: list
        Indices of the persistence forecast
    GCM_indices: list
        Indices of the GCM forecast
    sort_rows: bool
        Whether to sort the subset by the chosen colname before plotting
    savefile: str
        Name of the file to be written (default not saved)
    input_axes: obj (optinoal)
        Whether to use input axes for the plot
        Axes for the plot
    line06: bool
        Whether to plot the 0.6 line
    lstyle: dict
        Line style for the plot
    tstyle: dict
        Text style for the plot
    
    Returns
    =======
    None
    '''
    df = pd.read_csv(file)
    
    subset = df.loc[row_indices, 'M0':'M12']
    labels = df.loc[row_indices, colname]

    lstyle, tstyle = define_defaults_values(lstyle, tstyle)

    if sort_rows:
        combined = pd.concat([subset, labels.rename("Label")], axis=1)
        try:
            combined["Label"] = pd.to_numeric(combined["Label"])
        except ValueError:
            combined["Label"] = combined["Label"].astype(str)
        combined = combined.sort_values(by="Label")
        subset = combined.drop(columns="Label")
        labels = combined["Label"]


    if input_axes:
        ax = input_axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    
    for idx, row in subset.iterrows():
            ax.plot(row.index, row.values, label=labels.loc[idx])

  
    if per_indices is not None:
        for idx in per_indices:
            ax.plot(df.loc[idx, 'M0':'M12'], **{**lstyle, 'linestyle':'dashed'}, color='black')
    else:
        ax.plot(df.loc[1, 'M0':'M12'], label='Persistence', **{**lstyle, 'linestyle':'dashed'}, color='black')

    if GCM_indices is not None:
        for idx in GCM_indices:
            ax.plot(df.loc[idx, 'M0':'M12'], label='GCM', **{**lstyle, 'linestyle':'dashed'}, color='green')
    else:
            ax.plot(df.loc[2, 'M0':'M7'], label='GCM', linestyle='dashed', color='green')

    if line06:
        ax.axhline(0.6, linestyle='dashed', color='blue')
    ax.set_ylim([0.5,1.0])
    ax.set_xlabel(xlab, fontname='Futura')
    ax.set_ylabel(ylab, fontname='Futura')
    ax.legend(loc='best', fontsize='x-small')
    ax.grid(alpha=0.3)
    ax.set_title(title, **tstyle)
    
    if not input_axes:
        if savefile:
            plt.savefig(savefile, format='pdf')

        plt.show()
    return ax if input_axes else None

def boxplot(file, verify_dyn=False, input_axes=None, savefile=False, pltype='ACC'):
    '''
    Make a box plot of the data in a pickle file. It also
    write a tex files with the description of the plot

    Parameters
    ==========
    file: str
        Path to the pickle file
    verify_dyn: bool
        Whether to include the GCM-based forecast
    input_axes: bool
        Whether to use input axes for the plot
    savefile: str
        Name of the file to be written (default not saved)
    pltype: str
        Type of plot to be made (default ACC)

    Returns
    =======
    None
    '''
    # Set latex directory
    texpath = os.path.expanduser("~")+'/CMCC Dropbox/Antonio Navarra/AI'
    texdir = zai.create_subdirectory(texpath, 'LATEX')
    
    # Read pickle file
    with open(file, "rb") as f:
        dat=pickle.load(f)
    if dat['subcase'] == 203:
        istart = 3
    else:
        istart = 1

    #dat is a dictionary with the following keys
    #dict_keys(['description', 'subcase_var', 'subcase', 'dataversion',
    #  'invar_dict', 'pred_dict', 'author', 'date', 'name', 'config', 'area', 'size_model', 'best_val_loss', 'model_config', 
    # 'params', 'verification_field', 'input_fields', 'output_fields',
    #  'eof', 'GCMcorr', 'corr', 'pers'])
    
    
    # Create a box plot with customization
    if input_axes:
        ax = input_axes
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Extract relevant data from the dictionary
    if pltype == 'ACC':
        corrresult = dat['corr']
        persistence = dat['pers']
        if verify_dyn:
            corrdyn = dat['GCMcorr']
    elif pltype == 'RMS':
        corrresult = dat['rms']
        persistence = dat['rmsper']
        if verify_dyn:
            corrdyn = dat['GCMrms']
    else:
        raise ValueError("Invalid plot type. Choose 'ACC' or 'RMS'.")

    ntim, nens = corrresult.shape


    #Customizing box plot appearance
    
    # Create a box plot with different colors for each dataset
    box1 = ax.boxplot(corrresult[istart:,:].T, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='blue'),
                      medianprops=dict(color='darkblue',linewidth=2),
                      whiskerprops=dict(color='blue'),
                      capprops=dict(color='blue'),
                      flierprops=dict(markeredgecolor='blue'),
                      )
    # change here the xlabel to np.range(istart,ntim)
    

    box2 = ax.boxplot(persistence[istart:,:].T, patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', color='green'),
                      medianprops=dict(color='darkgreen',linewidth=2),
                      whiskerprops=dict(color='green'),
                      capprops=dict(color='green'),
                      flierprops=dict(markeredgecolor='green'),
                      )
    
    if verify_dyn:
        box3 = ax.boxplot(corrdyn[istart:,:].T, patch_artist=True, widths=0.25,
                      boxprops=dict(facecolor='lightpink', color='red'),
                      medianprops=dict(color='darkred',linewidth=2),
                      whiskerprops=dict(color='red'),
                      capprops=dict(color='red'),
                      flierprops=dict(markeredgecolor='red'),
                      )
   
    
    # Add a legend
    ax.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]], ['DeepSeason', 'Persistence', 'GCM'], loc='best', prop={'family': 'Futura'})
    
    # Define titleleft from the verification key in the dictionary
    titleleft = zai.transform_strings([dat['verification_field']])

    # Define titleright from the subcase key in the dictionary
    titleright = [str(dat['subcase'])]

    # Add title and labels
    ax.set_title(titleleft[0], loc='left', fontname='Futura')
    ax.set_xlabel('Forecast Month', fontname='Futura')
    ax.set_ylabel(pltype, fontname='Futura')
    
    ax.set_xlim([istart-0.5, ntim+0.5])

    if savefile:
        savefile = f"{texdir}/{file}_boxplot.tex"

    # Write to a tex file a latex piece describing the box plot
    tex_content = f"""
    \\documentclass{{article}}
    \\usepackage{{graphicx}}
    \\begin{{document}}
    \\section*{{Box Plot Description}}
    Correlation box plot for the verification field {titleleft[0]} and for the {dat['area']} region. 
    Lightblue is used for the model forecast, lightgreen for the persistence forecast, and lightpink for the GCM-based forecast. Median and quartile values
    are shown in dark blue, dark green, and dark red, respectively. The input fields used are {zai.transform_strings(dat['input_fields'])}. 
    In this case the EOF truncation used is {dat['eof']},  the size of the hidden space of the
    model is {dat['params']['D_DIM']} and the discount parameter for the loss function is {dat['params']['discount']}. 
    \\begin{{figure}}[h!]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{{savefile.replace('.tex', '.pdf')}}}
    \\caption{{Correlation box plot for the verification field {titleleft[0]} and for the {dat['area']} region. 
    Lightblue is used for the model forecast, lightgreen for the persistence forecast, and lightpink for the GCM-based forecast. Median and quartile values
    are shown in dark blue, dark green, and dark red, respectively. The input fields used are {dat['input_fields']} 
    and the LAGS parameters are {dat['params']['lags']}.
    In this case the EOF truncation used is {dat['eof']},  the size of the hidden space of the
    model is {dat['params']['D_DIM']} and the discount parameter for the loss function is {dat['params']['discount']}.}}
    \\end{{figure}}
    \\end{{document}}
    """

    lab = file
    with open(f"{texdir}/{lab}_boxplot.tex", "w") as tex_file:
        tex_file.write(tex_content)

    # Show the plot
    if not input_axes:
        if savefile:
            plt.savefig(savefile, format='pdf')

        plt.show()

    return ax if input_axes else None

def write_var_excel(name='Y620', varname='best_val_loss', vcases=[1, 2, 3, 4, 5], subcases=[8], filename=None, datain='_corr_data', write_latex=False):
    """
    Read several pickle files, extract the requested variable "varname" from each,
    and write them into a DataFrame with columns = subcases and rows = vcases.
    Optionally write the DataFrame to an Excel file and also output a LaTeX table.
    """
    texpath = os.path.expanduser("~") + '/CMCC Dropbox/Antonio Navarra/AI'
    texdir = zai.create_subdirectory(texpath, 'LATEX')
    
    # Decode subcase labels
    subcases_labels, _ = _decode_subcases(subcases)
    
    # Create a DataFrame with rows = vcases and columns = subcases_labels
    data_df = pd.DataFrame(index=vcases, columns=subcases_labels, dtype=float)
    
    # Fill DataFrame by reading each pickle file and extracting "varname"
    for i, sc in enumerate(subcases):
        col_label = subcases_labels[i]
        for vc in vcases:
            file_path = f"{name}_V{vc}_{sc}{datain}"
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)
            data_df.loc[vc, col_label] = data_dict[varname]
    
    # Write to Excel if requested
    if filename:
        data_df.to_excel(filename)

    # Optionally write LaTeX code for the table
    if write_latex:
        # df_mean = data_df.copy()
        # for col in df_mean.columns:
        if isinstance(data_dict[varname], int):
            # do not round integers
            latex_code = data_df.to_latex(float_format='%.f')
        else:
            latex_code = data_df.to_latex(float_format='%.2f')
          
        latex_filename = filename.replace('.xlsx', '.tex') if filename else 'results.tex'
        with open(f"{texdir}/{latex_filename}", 'w') as tex_file:
            tex_file.write(latex_code)

    return

def _decode_subcases(subcases):
    '''
    Return a list of labels for the subcases'
    '''
    subcases_labels = []
    
    for sc in subcases:
        # make a list of if statements for subcases
        if sc == 1:
            subcases_labels.append('1')
        elif sc == 2:
            subcases_labels.append('DISCOUNT')
            trial_labels = ['1.0', '0.9', '0.7', '0.3', '0.1']
        elif sc == 3:
            subcases_labels.append('HID')
            trial_labels = ['1024', '512', '256', '128', '64']
        elif sc == 4:    
            subcases_labels.append('LAGS')
            trial_labels = ['[1]', '[1, 2]', '[1, 2, 3]', '[1, 2, 3, 4]', '[1, 2, 3, 4, 5, 6]']
        elif sc == 5:
            subcases_labels.append('5')
            trial_labels = ['[10,25,15,25]', '[15,25,15,25]', '[25,25,25,25]', '[35,25,35,25]', '[45,25,45,25]']
        elif sc == 6:
            subcases_labels.append('LAYERS')
            trial_labels = [1,2,4,8,16]
        elif sc == 7:
            subcases_labels.append('7')
        elif sc == 8:
            subcases_labels.append('VARS')
            trial_labels = ['[SST]', '[SST, U850]', '[SST, U850,SP]', '[SST, U850,SP, OLR]', '[SST, OLR]']
        elif sc == 9:
            subcases_labels.append('HID')
            trial_labels = ['1024', '512', '256', '128', '64']
        elif sc == 10:
            subcases_labels.append('LAGS')
            trial_labels = ['[1]', '[1, 2]', '[1, 2, 3]', '[1, 2, 3, 4]', '[1, 2, 3, 4, 5, 6]']
        elif sc == 11:
            subcases_labels.append('LAYERS')
            trial_labels = [1,2,4,8,16]
        elif sc == 12:
            subcases_labels.append('EOF')
            trial_labels = ['[10,25]', '[15,25]', '[25,25]', '[45,25]', '[45,45]']
        elif sc == 13:
            subcases_labels.append('DISCOUNT')
            trial_labels = ['[1.0]', '[0.9]', '[0.7]', '[0.5]', '[0.1]']
        elif sc == 100:
            subcases_labels.append('DISCOUNT')
            trial_labels = ['EOF15']
        elif sc == 21:
            subcases_labels.append('VARS')
            trial_labels = ['[T2M]', '[T850, T2M]', '[U850, T2M]', '[SP, T2M]', '[U850, V850, T850, SST, T2M]']
        elif sc == 23:
            subcases_labels.append('EOF')
            trial_labels = ['[5,5]', '[6,6]', '[7,7]', '[8,8]', '[10,10]']
        elif sc == 24:
            subcases_labels.append('LAYERS')
            trial_labels = [1,2,4,8,12]
        elif sc == 25:
            subcases_labels.append('DISCOUNT')
            trial_labels = ['[1.0]', '[0.9]', '[0.7]', '[0.5]', '[0.1]']
        else:
            raise ValueError('Invalid subcase')
    
    return subcases_labels, trial_labels

def Two_Forecast_plots(F, V, field, level, cont, timestart, pictimes, figsize=(12, 8), world_projection='Atlantic', colorbars=True, maintitle='', picturelabels=None, arealatz=None, arealonz=None, centlon=0, labfile=None):
    '''
    Make verification plots for the field `field` and level `level` for the dataset `F` and `V`.
    Plots are made in two columns of `maxtime` rows each.
    
    Parameters
    ==========
    
    F: xarray dataset
        Dataset with the forecast to be analyzed
    V: xarray dataset
        Dataset with the verification field
    figsize: tuple
        Size of the figure
    world_projection: string
        World Projection for the plot
    maintitle: string
        Main label for the plot
    picturelabels: namedtuple
        List of labels for the pictures
    field: string
        Field to be analyzed
    level: string
        Level to be analyzed
    timestart: string
        Starting date for the plotting
    pictimes: list
        List of times to be plotted
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
    
    column = len(pictimes)
    maxV = len(V['time'])
    maxF = len(F['time'])

    title = maintitle + field if level is None else maintitle + field + level

    index = int(np.where(F['time'] == pd.Timestamp(timestart))[0]) 
    iv = index 

    fstyle = {'fontsize': 11, 'fontfamily': 'futura', 'fontweight': 'bold'}
    tstyle = {'fontsize': 12, 'fontfamily': 'futura', 'fontweight': 'bold'}
    sup_style = {'fontsize': 24, 'fontfamily': 'futura', 'fontweight': 'bold'}

    fig, ax, pro = zmap.init_figure(2, column, world_projection, constrained_layout=False, figsize=figsize)
    for k, i in enumerate(pictimes):
        if i <= maxF:
            label12 = str(F.time[i].data)[0:10]
            label13 = f'{picturelabels[0][k]}' if picturelabels is not None else f'Lead Month {i}'
            out = zmap.xmap(F.isel(time=i).unstack(), cont, pro, ax=ax[0, k], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                            xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                            custom=True, label_style=fstyle, title_style=tstyle,
                            title={'maintitle': 'Forecast', 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            out = _fix_labels(out, 'left')
            if colorbars:
                zmap.add_colorbar(fig, out.handles['filled'], ax[0, k], label_size=10, edges=True)
        else:
            ax[0, k].axis('off')

        if i <= maxV:
            label13 = f'{picturelabels[1][k]}' if picturelabels is not None else ''
            out2 = zmap.xmap(V.isel(time=i).unstack(), cont, pro, ax=ax[1, k], refline=None, c_format='{:4.2f}', data_cent_lon=centlon,
                            xlimit=(arealonz[0], arealonz[1]), ylimit=(arealatz[1], arealatz[0]), 
                            custom=True, label_style=fstyle, title_style=tstyle,
                            title={'maintitle': 'Obs', 'lefttitle': label12, 'righttitle': label13}, cmap='seismic', contour=False)
            out2 = _fix_labels(out2, 'right')
            if colorbars:
                zmap.add_colorbar(fig, out2.handles['filled'], ax[1, k], label_size=10, edges=True)
        else:
            ax[1, k].axis('off')

        # Fix properties of bounding box
        zmap.changebox(out.ax, 'all', linewidth=2, color='black', capstyle='round')
        zmap.changebox(out2.ax, 'all', linewidth=2, color='black', capstyle='round')

    if not colorbars:
        # Create an axis on the bottom of the figure for the colorbar 
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
        # Create a colorbar based on the first image
        cbar = fig.colorbar(out.handles['filled'], cax=cax, orientation='horizontal')
        # cbar.set_label('Colorbar Label')

    plt.suptitle(title, **sup_style)

    if labfile is not None:
        plt.savefig(labfile, orientation='landscape', format='pdf')

    plt.show()
    return ax

def write_var_table(name='Y620', case='LAGS', subcases=[8,9], rows=[0,3,5,7,9,11], srwox=[1,2],
                    filename='TableTex', skill=False,  stype='mean',mark='max'):
    '''
    Read several csv files, with "name" and several subcases, option for "skill cases", and write a latex table from
    input "rows", add the single rows in "special rows" and write the table in "filename"
    '''
    texpath = os.path.expanduser("~")+'/CMCC Dropbox/Antonio Navarra/AI'
    texdir = zai.create_subdirectory(texpath, 'LATEX')

    #Read several csv files looping over subcases with the skill scores
    #Write the rows in the latex file
    with open(f'{texdir}/{filename}', 'w') as texfile:
        
        header_written = False
        for sc in subcases:
            file = f'{skill}_Scores_{name}_{sc}_{stype}.csv'
            df = pd.read_csv(file)

            desired_headers = [case] + [f"M{i}" for i in range(13)]
           
            # Keep only the desired columns if they exist
            existing_cols = [c for c in desired_headers if c in df.columns]
            
            # Filter the rows we need
            df_filtered_main = df.iloc[rows][existing_cols]
            df_filtered_main = df_filtered_main.sort_values(by=case)
            df_filtered_special = df.iloc[srwox][existing_cols]
            df_filtered_special.iloc[0, 0] = "PERS"
            df_filtered_special.iloc[1, 0] = "GCM"
            # Combine them
            df_combined = pd.concat([df_filtered_main, df_filtered_special])
            
            # Replace NaN with blank
            df_combined.fillna('', inplace=True)
            
            # Color the maximum value in each M0..M12 column in red and center all entries
            columns_to_color = [f"M{i}" for i in range(1,13) if f"M{i}" in df_combined.columns]
            for col in columns_to_color:
                data = pd.to_numeric(df_combined[col], errors='coerce')
                if mark == "max":
                    max_val = data.max(skipna=True)
                else:
                    max_val = data.min(skipna=True)
                for idx in df_combined.index:
                    if pd.notnull(data.at[idx]) and data.at[idx] == max_val:
                        df_combined.at[idx, col] = f"\\textcolor{{red}}{{{data.at[idx]:.2f}}}"
                    elif pd.notnull(data.at[idx]):
                        df_combined.at[idx, col] = f"{data.at[idx]:.2f}"

            # Convert to LaTeX, ensuring we don't escape our LaTeX commands and center columns
            latex_table = df_combined.to_latex(float_format='%.2f',
                index=False, 
                header=not header_written, 
                na_rep='', 
                escape=False,
                column_format='c' * len(df_combined.columns)
            )
            texfile.write(latex_table + "\n")

            header_written = True
    return

def define_defaults_values(lstyle, tstyle):
    '''
    Define default values for the line and title styles
    '''
    default_line_style = {
        'linewidth': 1,
        'linestyle': '-',
    }
    if lstyle is None:
        lstyle = default_line_style
    else:
        lstyle = {**default_line_style, **lstyle}

    default_title_style = {
        'fontsize': 20,
        'fontfamily': 'futura',
        'fontweight': 'bold',
    }
    if tstyle is None:
        tstyle = default_title_style
    else:
        tstyle = {**default_title_style, **tstyle}

    return lstyle, tstyle

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

# suggest some methods to obtain the signficant values for correlation calculated using the
# xarray function .corr write the code adding also the code for the calculation of the p-values
def calculate_significance(corr, n, alpha=0.05):
    """
    Calculate the significance of correlation coefficients
    using the Fisher Z-transformation and mask non-significant values.

    Parameters
    ----------
    corr : xarray.DataArray
        Correlation coefficients.
    n : int
        Sample size.
    alpha : float, optional
        Significance level (default is 0.05).
        
    Returns
    -------
    significant_corr : xarray.DataArray
        Correlation coefficients with non-significant values masked as NaN.
    p_values : xarray.DataArray
        P-values for the correlation coefficients.
    """
    # Calculate Z-scores using Fisher Z-transformation
    z = 0.5 * np.log((1 + corr) / (1 - corr))
    
    # Calculate standard error
    se = 1 / np.sqrt(n - 3)
    
    # Calculate Z-scores for significance testing
    z_scores = z / se
    
    # Calculate p-values
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    # Mask non-significant values (retain sign of correlation)
    significant_corr = corr.where(p_values < alpha)
    
    return significant_corr, p_values



import xarray as xr
import numpy as np
from scipy import stats

def calculate_significance(corr: xr.DataArray,
                           n,
                           alpha: float = 0.05,
                           two_tailed: bool = True,
                           method: str = "fisher"):
    """
    Significance test for correlation coefficients.

    Parameters
    ----------
    corr : xr.DataArray          Correlation coefficients (−1 < r < 1)
    n    : int or xr.DataArray   Sample size(s) (must be > 3)
    alpha: float                 Significance level
    two_tailed : bool            Two‑tailed if True, else one‑tailed
    method : {"fisher","t"}      Approximate (normal) or exact (t) test
    """
    if (xr.where(n <= 3, True, False)).any():
        raise ValueError("Sample size n must be > 3 for a significance test")

    # Clip r to avoid log/0 overflow
    r = corr.clip(-0.999_999, 0.999_999)

    if method == "fisher":
        z   = np.arctanh(r)            # same as 0.5*log((1+r)/(1-r))
        se  = 1 / np.sqrt(n - 3)
        z_s = z / se
        p   = stats.norm.sf(np.abs(z_s))
    elif method == "t":
        t   = r * np.sqrt((n - 2) / (1 - r ** 2))
        p   = stats.t.sf(np.abs(t), n - 2)
    else:
        raise ValueError("method must be 'fisher' or 't'")

    if two_tailed:
        p *= 2

    sig_corr = r.where(p < alpha)

    return sig_corr, p
