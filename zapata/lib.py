import ipywidgets as widgets
import os
from IPython.display import IFrame,Javascript
# import pygrib
import numpy as np
import scipy.linalg as sc
import time
import matplotlib.pyplot as plt
import xarray as xr

def name_notebook(newname):
    '''
    Change name to Jupyterlab instance
    '''
    Javascript('document.title="{}"'.format(newname))


def _showpic(picture_file):
    """ Show `picture_file` in a widget."""

    with open(picture_file, "rb") as file:
        (d,type)=os.path.splitext(picture_file)
        image = file.read()
        fig=widgets.Image(
        value=image,
        format=type[1:3],
        layout=widgets.Layout(width='800px',height='900px'),)
        display(fig)

def _showpdf(file_loc):
    """ Show `file_loc` pdf file in a widget."""
    file_loc="./"+os.path.basename(file_loc)
    print(file_loc)
    display(IFrame(file_loc,width=800,height=800) )

def showfig(pic):
    """ Show picture in a widget."""   
    (dum,type)=os.path.splitext(pic)
    print(type)
    if type == '.pdf':
        _showpdf(pic)
    else:
        _showpic(pic)
        print('Displaying '+ pic)
    print('Stop')
    
def remove_values_from_list(the_list, val):
    """ Remove value `val` from list `the_list`"""
    return [value for value in the_list if value != val]

def makename(var,lev,yy,mm,dd):
    """ Utility to create names for ERA5 files. """
    return var + "_" + lev + "_" + str(yy) +"_"+ str(mm) + "_" + str(dd) + ".grb"

def makemm(var,lev,yy,mm):
    """ Utility to create names for ERA5 numpy files"""
    work1 = var + lev + '/'
    work2 = var + "_" + lev + "_" + str(yy) +"_"+ str(mm) +'_MM'  + ".npy"
    return work1 + work2

def makefilename(dir,var,lev,yy,mm,ext):
    """ Generalize file name creation """
    work1 = dir + '/'
    work2 = var + "_" + lev + "_" + str(yy) +"_"+ str(mm) + "." + ext
    return work1 + work2

def adddir(name,dir):
    """ Add `dir` directory name to `name` file"""
    return dir +'/' + name.split('.')[0]+'.npy'

def makedir(fndir):
    """Create Directory `fndir`"""
    try:
        # Create target Directory
        os.mkdir(fndir)
        print("Directory " , fndir ,  " Created ") 
    except FileExistsError:
        print("Directory " , fndir ,  " already exists")

def movefile(oldfile, newdir):
    """Move file from `oldfile` to `newdir`"""
# Move File 'oldfile' to directory 'newdir', with error control
    try:
        command =' mv ' + oldfile + ' ' + newdir  
        print(command)
        os.system(command)
    except: 
        print('Error in Moving Data Files... ',oldfile,'  to new directory .....', newdir)

def copyfile(oldfile, newdir):
    """Copy file from `oldfile` to `newdir`"""
# Move File 'oldfile' to directory 'newdir', with error control
    try:
        command =' cp ' + oldfile + ' ' + newdir  
        print(command)
        os.system(command)
    except: 
        print('Error in Copying Data Files... ',oldfile,'  to new directory .....', newdir)

def chop(a,epsilon=1e-10):
    """Eliminate real small complex number converting to real"""
    check=sc.norm(a.imag)
    if check < epsilon:
        out=a.real
    else:
        out=a
    return out

def year2date(years,i):
    """ Transform index i in string date yy/mm.
    
    Rounding requires the small shift
    Years are obtained from np.arange(1979,2018, 1/12)
    """
    mon=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    y=str(int(years[i]+0.001))
    m=np.mod(int(round((years[i]-int(years[i]))*12)),12)
    date = mon[m] + ' ' + y
    return date

def date2year(years,date):
    """Transform index date ['Jan' '1989' ] in index i.
    
    Years are from np.arange(1979,2018, 1/12)
    """
    mon=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    y=float(date[4:8])
    m=mon.index(str(date[0:3]))
    index = (y-1979)*12 + m
    return int(index)

def putna(left,right, xar, scalar = None):
    '''
    Put NaN in xarray according if they are laying in the interval `left,right`

    Parameters
    ==========

    left,right: 
        Extremes of the interval where the values must be NaN
    xar :   
        Xarray
    scalar :
        If set all entries not satisfying the condition are put equal to `scalar`

    Returns
    =======

    Modified array

    '''

    if scalar:
        out=scalar
    else:
        out=xar

    return xr.where((xar < right) & (xar > left), np.nan, out)

def go_to(dir):
    '''
    Set Working directory

    Parameters
    ==========

    dir:    
        Target directory relative to users' root directory

    YIELD
    =====

    Change working directory
    '''

    homedir = os.path.expanduser("~")

    print('Root Directory for Data ',homedir)
    #Working Directory
    wkdir =   homedir + '/'+ dir
    os.chdir(wkdir)
    print(f'Changed working directory to {wkdir}')
    return wkdir

def long_string(lon,cent_lon=0):
    '''
    Get nice formatted longitude string

    Parameters
    ==========
    lon:
        Longitude
    cent_lon:
        Central longitude for projection used

    Yield
    =====

    string in nice format
    '''
    E = 'E'
    W = 'W'
    if cent_lon == 0:
        if lon < 0:
            out = str(-lon) + W
        elif lon > 0:
            out = str(lon) + E
        else:
            out = str(lon)
    elif cent_lon == 180:
        if lon > 0:
            out = str(lon) + W
        elif lon < 0:
            out = str(-lon) + E
        else:
            out = str(lon)
    else:
        SystemError(f'Error in longitude string cent_lon {cent_lon} ')
    return out
def lat_string(lat):
    '''
    Get nice formatted latitude string

    Parameters
    ==========
    lat:
        Latitude

    Yield
    =====

    string in nice format
    '''
    
    if lat < 0:
        out = str(-lat) + 'S'
    elif lat > 0:
        out = str(lat) + 'N'
    else:
        out = 'Equator'
    
    return out



