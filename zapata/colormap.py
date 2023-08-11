'''
Color control and choice 
========================

Colormap Selection 
------------------
A number of colormap obtained from SciVis (`https://sciviscolor.org/`). 
The presently available colormap are shown below.

.. image:: ../resources/colormap.png
        :scale: 100 %
        :align: center

Use `make_cmap` to generate a matplotlib class colormap based on the
xml maps in the library above. `make_cmap` can also be used to generate
a matplotlib colormap custom made as a xml file.

Utilities 
---------


'''
import os
import sys
from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from lxml import etree

homedir = os.path.expanduser("~")
COLPATHDIR = homedir + '/Dropbox (CMCC)/GitHub/Zapata/zapata/SciVis_colormaps'

def make_cmap(xml,colpath=COLPATHDIR):
    '''
    | Convert colormap from `PARAVISION` and `SCIVISION` 
    | https://sciviscolor.org/outlier-focused-colormaps/
    
    Parameters
    ==========
    xml: str
        Name colormap in xml format 
    
    colpath: path
        Path to the location of xml colormap
    
    Returns
    =======
    colormap:    
        Colormap object
    '''

    print('Using colormap', colpath +'/'+ xml+'.xml')
    vals = load_xml(colpath +'/'+ xml+'.xml')
    colors = vals['color_vals']
    position = vals['data_vals']


    if len(position) != len(colors):
        sys.exit('position length must be the same as colors')

    cdict = {'red':[], 'green':[], 'blue':[]}

    if position[0] != 0:
        cdict['red'].append((0, colors[0][0], colors[0][0]))
        cdict['green'].append((0, colors[0][1], colors[0][1]))
        cdict['blue'].append((0, colors[0][2], colors[0][2]))
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    if position[-1] != 1:
        cdict['red'].append((1, colors[-1][0], colors[-1][0]))
        cdict['green'].append((1, colors[-1][1], colors[-1][1]))
        cdict['blue'].append((1, colors[-1][2], colors[-1][2]))
    cmap = col.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def load_xml(xml):
    '''
    Load colormap in `xml` format

    Parameters
    ==========
    xml: str
        Name colormap in xml format 
    
    Returns
    =======
    dict :    
        colorvals   Colorvalues
        data_vals   Color scale

    '''
    try:
        xmldoc = etree.parse(xml)
    except IOError as e:
        print ('The input file is invalid. It must be a colormap xml file. Go to https://sciviscolor.org/home/colormaps/ for some good options')
        print ('Go to https://sciviscolor.org/matlab-matplotlib-pv44/ for an example use of this script.')
        sys.exit()
    
    data_vals=[]
    color_vals=[]
    for s in xmldoc.getroot().findall('.//Point'):
        data_vals.append(float(s.attrib['x']))
        color_vals.append((float(s.attrib['r']),float(s.attrib['g']),float(s.attrib['b'])))
    return {'color_vals':color_vals, 'data_vals':data_vals}

def plot_cmap(colormap):
    '''
    Show colormap
    '''

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig=plt.figure(figsize=(8,1))
    map=fig.add_subplot(111)
    map.set_frame_on(False)
    map.get_xaxis().set_visible(False)
    plt.title(colormap.name)
    map.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    map.imshow(gradient, aspect='auto', cmap=plt.get_cmap(colormap))
    plt.show(fig)
    return

def viewcmap(DIR):
    '''
    Show colormaps in directory `DIR`

    Colormaps are in `xml` format
    '''
    for i in sorted(os.listdir(DIR)):
        filename, file_extension = os.path.splitext(i)
        if file_extension == '.xml':
            tt=make_cmap(COLOR +'/'+ i)
            tt.name = filename
            plot_cmap(tt)
    return
def _showcolormap(COLOR,TGTDIR):
    '''
    Create picture of all colormap in directory COLOR
    and puts picture in directory TGTDIR
    '''
    fil = []
    for i in sorted(os.listdir(COLOR)):
        filename, file_extension = os.path.splitext(i)
        if file_extension == '.xml':
            fil.append(i)
    nplot=len(fil)
    nplot2 = int(nplot/2)
    fig,ax=plt.subplots(nrows=int(nplot/2),ncols=2,figsize=(12,32))
    for i in range(0,int(nplot/2)):
            filename, file_extension = os.path.splitext(fil[i])
            tt=make_cmap(filename,colpath=COLOR)            
            tt.name = filename
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
    
            axs = ax[i,0]
            axs.set_frame_on(False)
            axs.get_xaxis().set_visible(False)
            axs.set_title(tt.name)
            axs.get_yaxis().set_visible(False)
            axs.imshow(gradient, aspect='auto', cmap=tt)
    for i in range(0,int(nplot/2)):
            filename, file_extension = os.path.splitext(fil[i+nplot2])
            tt=make_cmap(filename,colpath=COLOR)           
            tt.name = filename
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
    
            axs = ax[i,1]
            axs.set_frame_on(False)
            axs.get_xaxis().set_visible(False)
            axs.set_title(tt.name)
            axs.get_yaxis().set_visible(False)
            axs.imshow(gradient, aspect='auto', cmap=tt)
    fig.tight_layout(pad=0.5)
    fig.show()
    plt.savefig(TGTDIR + '/colormap.png')
    return