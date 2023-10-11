'''
Contouring, stremlines and vector plots
=======================================

Latitude-Longitude Plotting 
---------------------------
A set of routines for horizontal plotting

**xmap, xsmap, xstmap, vecmap**

Zonal Plotting
--------------
These routine split zonal sections in pressures and latitude.

**zonal_plot, zonal_stream_plot, ocean_section_plot**

Utilities 
---------
**choose_contour, add_colorbar, make_ticks, set_title_and_labels**

Detailed Description:
---------------------

'''
import math as mp
from operator import is_
import sys
import cartopy.crs as car
import cartopy.util as utl

from scipy import integrate
import numpy as np

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt 
import matplotlib.path as mpath

# Set tight bounding box
plt.rcParams["savefig.bbox"] = 'tight'
# Set True Type Fonts
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42

import numpy as np
import xarray as xr

import cartopy.mpl.ticker as ctick
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import zapata.data as era
import zapata.computation as zcom
import zapata.lib as zlib

import mpl_toolkits.axes_grid1 as tl


import docrep
# Uses DOCREP for avoiding copying docstring, contrary to the docs delete works
# only on one param at the time.

d = docrep.DocstringProcessor()


def init_figure(rows,cols,proview,constrained_layout=True,figsize=(16,8),**kwargs):
    """
    Initialization for the entire figure, choose projection and number of panels.

    Parameters
    ----------
    rows :  
        Number of rows in multiple picture
    cols :       
        Number of columns in multiple picture
    figsize :    
        Size of figure in inches (height, width)
    proview :    
        Projection for the map

        * String:
            - *"Pacific"*,   cartopy PlateCarree, central longitude 180   
            - *"Atlantic"*,  cartopy PlateCarree, central longitude 0      
            - *"NHStereoEurope"*,  NH Stereo, central longitude 0     
            - *"NHStereoAmerica"*,  NH Stereo, central longitude 90   
            - *"SHStereoAfrica"*,  NH Stereo, central longitude 0
            - *"RobPacific"*,   Robinson, central longitude 180   
            - *"RobAtlantic"*,  Robinson, central longitude 0   

        * Dictionary:
            For projections requiring more parameters. The first element
            in the dictionary is the name of the projection followed by the parameters required

            -  projection: 'Satellite':
                - centlon:  Central Longitude
                - centlat:  Central Latitude
                - height:   Altitude of the view (m)

    constrained_layout :    
        True/False 

    Returns
    -------
    fig :   handle   
        Dictionary with matplotlib-like info on the plot    
    ax :     axes   
        Axes of the panels  
    pro :    projection
        Projection chosen   
    """
   
    # Choose Projection
    projection = choose_projection(proview)

    fig, ax = plt.subplots(rows, cols,figsize=figsize, constrained_layout=constrained_layout , \
                           subplot_kw={ "projection": projection},**kwargs)
    
    print(' Opening figure , %i rows and %i cols \n' % (rows,cols))
    
    return fig,ax,projection

def choose_projection(proview):
    '''
    Auxiliary routine to select projection

    Parameters
    ----------
    proview :    
        Projection for the map

        * String:
            - *"Pacific"*,   cartopy PlateCarree, central longitude 180   
            - *"Atlantic"*,  cartopy PlateCarree, central longitude 0      
            - *"NHStereoEurope"*,  NH Stereo, central longitude 0     
            - *"NHStereoAmerica"*,  NH Stereo, central longitude 90   
            - *"SHStereoAfrica"*,  NH Stereo, central longitude 0
            - *"RobPacific"*,   Robinson, central longitude 180   
            - *"RobAtlantic"*,  Robinson, central longitude 0   

        * Dictionary:
            For projections requiring more parameters. The first element
            in the dictionary is the name of the projection followed by the parameters required

            -  projection: 'Satellite':
                - centlon:  Central Longitude
                - centlat:  Central Latitude
                - height:   Altitude of the view (m)


    Returns
    -------
    projection :     
        Projection object   
    '''

#Defaults for Satellite
    default_satellite = {'centlon': 0.0, 'centlat': 0.0, 'z': 35785831}
   
    # This fixes the projection for the mapping: 
    #      Data Projection is then fixed in mapping routine
    # Select Projection parameters
    if isinstance(proview,str):
        if   proview == 'Pacific':
            projection = car.PlateCarree(central_longitude=180.)
        elif proview == 'Atlantic':
            projection = car.PlateCarree(central_longitude=0.)
        elif proview == 'NHStereoEurope':
            projection = car.NorthPolarStereo(central_longitude=0.)
        elif proview == 'NHStereoAmerica':
            projection = car.NorthPolarStereo(central_longitude=-90.)
        elif proview == 'SHStereoAfrica':
            projection = car.SouthPolarStereo(central_longitude=90.)
        elif proview == 'RobAtlantic':
            projection = car.Robinson(central_longitude=0.)
        elif proview == 'RobPacific':
            projection = car.Robinson(central_longitude=180.)
    elif isinstance(proview,dict):
        temp = { **default_satellite,**proview}
        print(temp['z'])
        if temp['projection'] == 'Satellite':
            # projection = car.NearsidePerspective(central_longitude=temp['centlon'],\
            #         central_latitude=temp['centlat'], satellite_height=['z']) 
            projection = car.NearsidePerspective(central_longitude=temp['centlon'],\
                    central_latitude=temp['centlat'], satellite_height=temp['z'])        
    else:
        print(' Error in init_figure projection {}'.format(proview))
        raise SystemExit
    
    #Store View in projection
    projection.view = proview
    
    return projection


@d.get_sections(base='xmap')
@d.dedent
def xmap(data, cont, prol, ax=None, fill=True,contour=True, \
                    data_cent_lon=180, \
                    clabel=True, c_format = ' {:6.0f} ', \
                    lonlabel=None,latlabel=None,\
                    refline=None, \
                    title={}, title_style={},\
                    xlimit=None,ylimit=None,\
                    cyclic=False,\
                    colorbar=False,cmap='coolwarm',\
                    coasts=True,color_land='lightgray',\
                    quiet=True,\
                    label_style={}):
    """
    Mapping function based on cartopy and matplotlib.

    The data is supposed to be on a Plate (lat-lon) projection and the central longitude can be defined 
    via the paramater `data_cent_lon`. The defualt is `data_cent_lon=180`, meaning that the central longitude is over the pacific. 
    Coordinates that go from 0 to 360 implicitly assume such a Pacific central longitude.

    For the `Pacific` view it is assumed that the central longitude is at the daetline
    whereas for the `Atlantic` view the central longitude is at Greenwich.

    The projection needs to be established in `init_figure`


    Parameters
    ----------
    field :
        xarray  --  cyclic point added in the routine
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    prol :
        Map Projection as initialized by init or it can be
        a string, one of the options in `figinit`
    data_cent_lon :
        Central longitude for the data projection
    latlabel:
        Position of Latitude grids
    lonlabel:
        Position of Longitude grids
    ax :
        Plot axis to be used   
    fill :
        True/False flag to have filled contours or not 
    contour :  
        True/False flag to have  contours or not
    refline :
        If a numeric value a single enhanced contour is plotted here
    clabel :
        True/False flag to have labelled contours or not
    c_format :
        Format for the contour labels    
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels
    cmap :
        Colormap
    cyclic:
        True/False Add or not cylic longitude
    coasts:
        False/True   Plotting or not empty coastlines 
    color_land:
        if coasts=False, use color_land for land
    xlimit:
        Longitude limits of the map, if not set they are derived from the data.
        The projection used is the geographical projection `pro`
    ylimit:
        Latitude limits of the map, if not set they are derived from the data.
        The projection used is the geographical projection `pro`
    quiet:
        (False)  Suppress all output
    
    
    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """
    
    #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }
    defaultLatLabel = [-60.,-30.,0.,30.,60.]
    defaultLonLabel = np.arange(-180,181,30)

    if label_style is not None:
        opt_label =  label_style
    else:
        opt_label =  defaultGridLabels
    
    if title is not None:
        title_text = title
    else:
        title_text =  defaultTitle

    if title_style is not None:
        opt_title =  title_style
    else:
        opt_title =  defaultTitleStyle
    
    if latlabel is not None:
        opt_latlabel =  latlabel
    else:
        opt_latlabel =  defaultLatLabel
    
    if lonlabel is not None:
        opt_lonlabel =  lonlabel
    else:
        opt_lonlabel =  defaultLonLabel

    #Projection
    if isinstance(prol,str):
        pro = choose_projection(prol)
        ax.projection = pro
    else:
        pro = prol
    this = pro.__class__.__name__

    # Set data projection
    data_proj = car.PlateCarree(central_longitude=data_cent_lon)
    
    #Add Cyclic point
    if cyclic:
        data = add_cyclic_lon(data)

    #Eliminate extra dimensions
    if len(data.shape) > 2:
        data = data.squeeze()
        
    # Add coastlines
    if coasts:
        ax.coastlines(linewidth=0.5)
    else:
        ax.coastlines(linewidths=0.5)
        ax.add_feature(cfeature.LAND, facecolor=color_land)
    
    if this  in  ['PlateCarree']:
        if ylimit is  None:
            ylim = ax.projection.y_limits
        else:
            ylim=ylimit
        ax.set_ylim(ylim)

        if xlimit is  None:
            xlim = ax.projection.x_limits
        else:
            xlim=xlimit
        ax.set_xlim(xlim)
    elif this in ['NorthPolarStereo','SouthPolarStereo']:
        if xlimit is  None:
            xlim = [np.amin(data.lon.values)-180,np.amax(data.lon.values)-180+0.001]
        else:
            xlim=xlimit
            
        if ylimit is  None:
            ylim = [np.amin(data.lat.values),np.amax(data.lat.values)]
        else:
            ylim=ylimit
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )
        
        ax.set_extent(list(xlim+ylim),car.PlateCarree())  # 
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.spines['geo'].set_linewidth(2.0)
    else:
        print(' Projection in `xmap` {}'.format(this))

    if not quiet:
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )

    #Choose Contours
    vmin = data.min()
    vmax = data.max()

    cc = choose_contour(vmin,vmax,cont)
    if not quiet:
        print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
    if cc is None:
        print(f'******** Constant field ***********')
        return None


    handles = dict()
    if fill:
        handles['filled'] = ax.contourf(data.lon, data.lat, data, levels=cc, cmap=cmap, transform=data_proj)
    if contour:
        handles['contours'] = ax.contour(data.lon, data.lat, data, levels=cc, \
            linewidths=0.8, colors='black',transform=data_proj)
    if refline != None:
        if type(refline) is list:
            refs = refline
        else:
            refs = [refline]
        handles['refline'] = ax.contour(data.lon, data.lat, data, levels=refs, linewidths=1.0, colors='black', transform=data_proj)
    
    
    # Label the contours
    if clabel and contour:
        ax.clabel(handles['contours'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
        if refline:
            ax.clabel(handles['refline'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, fmt=c_format.format )
    
    # Add gridlines
    label_style = opt_label
    gl = ax.gridlines(draw_labels=True, dms=True,xlabel_style=label_style, ylabel_style=label_style, \
                  linewidth=0.6, color='gray', alpha=1.0,linestyle='--', \
                  formatter_kwargs={'degree_symbol':''})
    gl.right_labels = True
    gl.top_labels = False
    gl.xlocator = mticker.FixedLocator(opt_lonlabel)
    gl.ylocator = mticker.FixedLocator(opt_latlabel)
   
    set_titles_and_labels(ax,title_text,{'xlabel':'','ylabel':''},opt_title,opt_label)
    
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(handles['filled'], cax=cax,orientation='vertical')
    
    return handles  

@d.get_sections(base='zonal_plot')
@d.dedent
def zonal_plot(data,ax,cont,cmap,colorbar=True, title={}, title_style={},\
                label_style={},\
                refline=None):
    """
    Zonal mapping function for xarray (lat,pressure). 
    
    Parameters
    ----------
    data :    
            xarray  --  cyclic point added in the routine (latitude, pressure)      
    cont :
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn] 
        * n                      n contours 
        * []                     automatic choice  
    ax :            
            Plot axis to be used      
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels  
    cmap:  
            Colormap     
    refline:   
            False/True if a zero line is desired     
    colorbar:   
            False/True if a colorbar is desired
    
    Returns
    --------
    
    handle: 
        Dict with plot parameters
    
    Examples
    --------
    
    >>> zonal_plot(data,ax,[],'BYR',colorbar=True, titel={'maintitle': 'Title', 'lefttitle': None, 'righttitle':None},refline=None)
    """

     #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }

   
    opt_label =  {**defaultGridLabels, **label_style}
    title_text =  {**defaultTitle, **title}
    opt_title =  {**defaultTitleStyle, **title_style}
    
    
    #Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
   

    handle = data.plot.contourf(
        ax=ax,                            # this is the axes we want to plot to
        cmap=cmap,                        # our special colormap
        levels=cc,      # contour levels specified outside this function
        xticks=np.arange(-90, 91, 15),  # nice x ticks
        yticks=[1000,850,700,500,300,200,100],    # nice y ticks
        add_colorbar=colorbar,               # don't add individual colorbars for each plot call
        add_labels=False                 # turn off xarray's automatic Lat, lon labels
    )
    if refline:
        hc = data.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.25,
        add_labels=False  # again turn off automatic labels
        )
    lev=data.pressure.values
    nlev=len(lev)
    ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    ax.set_xlim(90,-90)  # Invert x axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_update_ticks))
    

    # Add gridlines
    set_titles_and_labels(ax,title_text,{'xlabel':'Latitude','ylabel':'Pressure'},opt_title,opt_label)
   
    return handle

def zonal_stream_plot(datau, datav, ax, C=None,  \
                    lw=1, density=2,\
                    title={}, title_style={},label_style={},\
                    colorbar=False,cscale=None, cmap='coolwarm',norm=None, \
                    quiet=True):

    """
    Plot zonal streamline for fielddatu e datav.

    Parameters      
    ----------

    datau : xarray
        X component of the streamlines   
    datav : xarray
        Y component of the streamlines
    C :   
        Color of the stremalines ('black'). If it is xarray color the streamlines with the colormap ``cmap``       
    density :    
        Density of the streamlines      
    ax :          
        Plot axis to be used        
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels         
    cmap:  
        Colormap       
    smooth:   
        False/True if smoothing is desired
            
    colorbar:   
        False/True if a colorbar is desired      
    

    """
    # Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }

    xticks = np.arange(-90, 91, 15),  # nice x ticks
    yticks = [1000,850,700,500,300,200,100], 

    opt_label =  {**defaultGridLabels, **label_style}
    title_text =  {**defaultTitle, **title}
    opt_title =  {**defaultTitleStyle, **title_style}
   
    
    #select color scale
    this = type(C).__name__
    if this == 'str':
        color_scale=C
    elif this == 'DataArray':
        print(this)
        color_scale = C.interp(pressure=np.arange(100,1000,50)).data
    elif this == 'ndarray':
        color_scale=C
    else:
        color_scale='black'
        colorbar=False

    #Eliminate extra dimensions
    if len(datau.shape) > 2:
        datau = datau.squeeze()
        datav = datav.squeeze()
    #Interpolate on a pressure regular grid
    U=datau.interp(pressure=np.arange(100,1000,50))
    V=datav.interp(pressure=np.arange(100,1000,50))
   
    # Stream-plot the data
    # There is no Xarray streamplot function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    hc=ax.streamplot(U.lat.data, V.pressure.data, U.data, V.data, linewidth=1, density=density, color=color_scale, \
                     zorder=1,cmap=cmap   )
    #  Label the contours
    #     ax.clabel
    #         handles["contour"], fontsize=8, fmt="%.0f",  # Turn off decimal points
    #    )

    lev=U.pressure.values
    nlev=len(lev)
    ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_update_ticks))
    
     # Add gridlines
    set_titles_and_labels(ax,title_text,{'xlabel':'Latitude','ylabel':'Pressure'},opt_title,opt_label)
     
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(hc.lines, cax=cax,orientation='vertical')

    return hc

def choose_contour(vmin,vmax,cont):
    """
    Choose contours according to length of `cont`.

    Parameters
    -----------
    vmin :
        Max Value of the field
    vmax :
        Min Value of the field
    cont :  
        * [cmin,cmax,cinc]       fixed increment from cmin to cmax step cinc 
        * [ c1,c2, ..., cn]      fixed contours at [ c1,c2, ..., cn]  
        * n                      n contours 
        * []                     automatic choice   

    Returns
    --------
    
    contour_levels :    
        Levels of selected Contours

    Examples
    --------
    >>> [0,1000,10] # Contours from 0 1000
    >>> [0.01, 0.05, 1.0, 2.0, 5.0, 10.0] # Set Contours Leves   
    """
    if len(cont) == 3:
        print("Setting Fixed Contours")
        cc=np.arange(cont[0]-cont[2],cont[1]+2*cont[2],cont[2])
        print(' Contouring from ', cont[0], '  to', cont[1],' with interval  ',cont[2])  
    elif len(cont)> 3:
        cc=cont
        print('Fixed Contours to..',cont)
    elif len(cont) == 1:
        cc=cont[0]
        print('Number of Contours ',cc)
    else:
        cc=10
        #delta = abs(vmax.data-vmin.data)
        if mp.isclose(vmax.data, vmin.data):
            cc = 1
            print(f'Almost constant field {vmax.data}, {vmin.data}, one contours')
        else:
            print(f'Ten Contours automatic')
    return cc

def add_colorbar(fig, handle, ax, colorbar_size=0.05,label_size=10,edges=True):
    """
    Add colorbar to plot.
    
    Parameters
    -----------
    
    handle :   
        Handle to plot
    ax :   
        Axis to which to add the colorbar
    colorbar_size:    
        Size of colorbar as a fraction of the axis
    label_size:   
        Size of labels
    edges:   
        Draw edges of the color bar
    """    
    #Check for absence of plots
    if handle is None:
        return
        
    #Eliminate overlapping labels
    stride = 1
    if len(handle.levels) > 10:
        stride = int(len(handle.levels)/10)
    divider = tl.make_axes_locatable(ax)
    cax = divider.append_axes('bottom',size="3.5%", pad=0.4, axes_class=plt.Axes)
    ax.get_figure().colorbar(handle, cax=cax, orientation='horizontal',\
                        ticks=handle.levels[::stride],fraction=colorbar_size,drawedges=edges)
    cax.tick_params(labelsize=label_size)
    return
def make_ticks(xt,dt=20,quiet=True):
    '''
    Calculates nice tickes for the axes.
    Selecting 20 or 10 intervals

    Parameters
    ----------

    xt : 
        Axes limit [west, east]
    
    dt:
        Tick spacing

    Returns
    -------

    ticks:
        Nice ticks position
    quiet:
        T/F get Output
    '''
    n = (xt[1]-xt[0])/dt
    if not quiet:
        print(f' {n} Ticks set at {dt}   intervals') 
    return np.linspace(xt[0], xt[1], int(n+1))

def set_titles_and_labels(ax, title_text, label_text, opt_title,opt_label):
    """
    Utility function to handle axis titles, left/right aligned titles.

    Parameters  
    ----------

    ax :    
        Current axes to the current figure  
    
    title_text : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_text:
        * xlabel -- Label for the x axis
        * ylabel -- Label for the y axis
    label_style : dict
        Dictionary with the style for the labels  

    """
    
    ax.grid( linewidth=0.6, color='gray', alpha=1.0,linestyle='--') 
    ax.set_title(title_text['lefttitle'],  loc='left',**opt_label )
    ax.set_title(title_text['righttitle'],  loc='right',**opt_label )
    ax.set_title(title_text['maintitle'],  loc='center',**opt_title )

    ax.set_xlabel(label_text['xlabel'], **opt_label)
    ax.set_ylabel(label_text['ylabel'], **opt_label)

    return

d.delete_params('zonal_plot.parameters','data')
@d.dedent
def ocean_section_plot(data,ax,cont,cmap,colorbar=True, \
    fill=True,contour=True, \
    clabel=True, c_format = ' {:4.0f} ', \
    dxtick = 20, dytick = 10,\
    title={},title_style={},label_style={},\
          xlimit=None,ylimit=None,refline=None,view='Atlantic'):
    """
    Ocean Section mapping function for xarray (`lat,lon` ,depth). 
    
    Parameters
    ----------
    data :    
        xarray  --   latitude or longitude, depth)   
    %(zonal_plot.parameters.no_data)s
    fill :
        True/False flag to have filled contours or not 
    contour :  
        True/False flag to have  contours or not
    refline : 
        If a numeric value inlist form a single enhanced contour is pltted here
    clabel :
        True/False flag to have labelled contours or not
    c_format :
        Format for the contour labels
    dxtick : 
        Tick interval for longitudes
    dytick :
        Tick interval for latitudes   
    xlimit:
        x axis limits
        A Pacific view is obtained by giving limits in coordinae with
        the central longitude at Greenwich, e.g.  (120,-120) will produce a
        shifted Pacific section 
    ylimit:
        y axis limits
    view:
        Shift between `Atlantic` and `Pacific` views

    Returns
    -------
    
    handle: 
        Dict with plot parameters
    
    Examples
    --------
    
    >>> ocean_zonal_plot(data,ax,[],'BYR',colorbar=True, maintitle=None, lefttitle=None, righttitle=None,refline=None)
    """
    
    #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }

   
    opt_label =  {**defaultGridLabels, **label_style}
    title_text =  {**defaultTitle, **title}
    opt_title =  {**defaultTitleStyle, **title_style}
   
    
    #Set a working view
    work = data

    #Choose Contours
    vmin = data.min()
    vmax = data.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
        
    #Vertical levels
    lev=data.deptht.data
    nlev=len(lev)

    # Vertical ticks
    if lev.max() > 1500:
        levtick =[i for i in range(0,1000,200)] + [i for i in range(1000,5500,500)]
    elif lev.max() > 100:
        levtick =[i for i in range(0,1000,100)] 
    else:
        levtick = [i for i in range(0,1000,100)] 
    
    # Choose labels
    if 'lat' in data.dims:
        xlab = 'Latitude'
        xtick_loc = np.arange(-90,90,dytick)
        xtick_lab = [zlib.lat_string(i) for i in xtick_loc]
    elif 'lon' in data.dims:
        #Check longitude zero line
        if view == 'Pacific':
            #Pacific View
            print(f'Shifting Coordinates for Pacific View')
            work=work.roll(lon=-720,roll_coords=False)
            xlab = 'Longitude'
            xtick_loc = np.arange(-180,181,dxtick)
            dateline = int(len(xtick_loc)/2)
            xtick_lab = [zlib.long_string((i+180)%360) for i in xtick_loc]
            xtick_lab[dateline:] = [zlib.long_string(i -180) for i in xtick_loc[dateline:]]
        else:
            xlab = 'Longitude'
            xtick_loc = np.arange(-180,181,dxtick)
            xtick_lab = [zlib.long_string(i) for i in xtick_loc]
    else:
        xlab = 'Missing'
    
    handles = dict()
    if fill:
        handles['filled'] = work.plot.contourf(
            ax=ax,                            # this is the axes we want to plot to
            cmap=cmap,                        # our special colormap
            levels=cc,      # contour levels specified outside this function
            xticks=xtick_loc,  # nice x ticks
            yticks = levtick,    # nice y ticks
            add_colorbar=colorbar,               # don't add individual colorbars for each plot call
            add_labels=False                 # turn off xarray's automatic Lat, lon labels
        )
    if contour:
        handles['contours'] = work.plot.contour(
            ax=ax,                            # this is the axes we want to plot to
            colors='black',                       # our special colormap
            linestyles="-",
            linewidths=0.8,
            levels=cc,      # contour levels specified outside this function
            add_labels = False
        )
    if refline:
        handles["refline"] = work.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.0,
        add_labels=False  # again turn off autransform=car.Geodetic()tomatic labels
        )
    
    ax.set_xticks(xtick_loc)
    ax.set_xticklabels(xtick_lab)

    # Y limits
    if ylimit == None :
        ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    else:
        ax.set_ylim(ylimit)
    #X limits
    if (xlimit == None) and (xlab == 'Latitude'):
        ax.set_xlim(90,-90)  # Invert x axis
    elif (xlimit == None) and (xlab == 'Longitude'):
        ax.set_xlim(-180,180)  
    else:
        ax.set_xlim(xlimit)

    # Label the contours
    if clabel and contour:
        ax.clabel(handles['contours'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8,rightside_up=True,fmt=c_format.format )
        if refline:
            ax.clabel(handles['refline'], colors='black', inline= True, use_clabeltext=True, inline_spacing=5,
                fontsize=8, rightside_up=True, fmt=c_format.format )
    
    # set bathymetry to black
    ax.set_facecolor('k')
    
    set_titles_and_labels(ax,title_text,{'xlabel':xlab,'ylabel':'Depth'},opt_title,opt_label)
   
    return handles
def small_map(fig,ax,cent_lon=0,lat=None, lon=None):
    '''
    Add small map for sections. The line is drawn at one
    `lat` or `lon`.

    Parameters
    ==========

    fig:
        `fig` to add the small map
    ax:
        `ax` to used
    cent_lon:
        central longitude for longitude sections
    lat: (None)
        Latitude of the longitude section
    lon: (None)
        Longitude of the latitude section
    '''
    pro = car.PlateCarree(central_longitude=cent_lon)
    X,Y = ax.get_position().get_points()
    r_ax = fig.add_axes([0.125,0.125,Y[0]/8,Y[1]/8], anchor='SW', facecolor='w',projection=pro)
    lim = ax.get_xlim()

    r_ax.set_global()
    if lat is not None:
        r_ax.plot(lim,[lat,lat],transform=pro,ls='-',color='r')
    if lon is not None:
        r_ax.plot([lon,lon],lim,transform=pro,ls='-',color='r')
    r_ax.coastlines(resolution='110m')
    r_ax.gridlines()
    return

def adjust_data_centlon(data,centlon_old=0, centlon_new=180.,roll_coords=False):
    '''
    Adjust xarray to a different central longitude.
    The data are rolled without rolling the coordinates.

    Parameters
    ==========
    data: xarray
        xarray data to be shifed. it is assumed to have dimension `lat` and `lon`
    centlon_old:
        Original Central longitude of the data
    centlon_new:
        New central longitude of the data
    
    Returns
    =======
        Rolled xarray
    '''

    # Check longitude dimension

    if 'lon' not in data.dims:
        SystemError(f'Wrong dimension in `adjust_data_centlon` --> {data.dims}')
    #Check shift
    if abs(centlon_new-centlon_old) < 180:
        SystemError(f'Wrong dimension in `adjust_data_centlon` /n  \
        Actual implementation only for 180 shifts  --> {centlon_new} -- {centlon_old}')
    
    print(f'Shifting from {centlon_old} to {centlon_new}')

    #Get shift value
    shift = int(data.sizes['lon']/2+1)
    
    #Roll data to new cent_lon
    new = data.roll(lon=int(len(data.lon)/2),roll_coords=roll_coords)

    return new

def lon_sect(fig, ax, field, lon_sec,labelR=[],maintit=[],lmax=1000, xl=None,cmap='coolwarm', \
             contour=True,levels=[], sharp=0., **kw):
    '''
    Latitude section at a fixed longitude.

    This routine performs a section at the chosen longutude. The `sharpness` of he section is controlled by the parameter `sharp`. 
    A `sharp = 0` indicates a precise section at the nominal longitude, where any value different from zero will result
    in average on an interval of length 2*sharpness centered at the longitude.

    Parameters
    ==========
    
    fig:
        Figure to be used
    ax :
        Plot axis to be used
    field: xarray
        Data to be plotted
    lon_sec:
        Longitude of the section
    lmax:
        Maximum depth of the section
    xl:
        Extent of the section
    cmap:
        colormap
    contour:
        True/false for having contours
    levels:
        Contour Levels ( see `choose_contour`)
    sharp: positive float
        Semi-width of the averaging interval around the section.
        The average ignore NaN and therefore small features can be lost in the averaging.
    labelR:
        Label ont he top roght of the plot
    maintit:
        Main title
    
    Returns
    =======

    handle:
        handle to the plot
    '''
    
    #lon_sec = -140
    lev_min = 0
    lev_max = lmax
    #
    label2= zlib.long_string(lon_sec) + ' Longitude'
    title = {'maintitle':maintit,'lefttitle':label2,'righttitle':lanelR}
    # Check sharpness
    if sharp > 0:
        T0 = field.sel(deptht=slice(lev_min,lev_max),lon=slice(lon_sec-sharp,lon_sec+sharp)).mean(dim='lon')
    elif sharp < 0:
        raise SystemExit(f'Wrong sharp in lon_sect --> {sharp}')
    else:
        try:
            T0 = field.sel(deptht=slice(lev_min,lev_max),lon=lon_sec)
        except:
            raise SystemExit(f'Longitude missing in data set  --> {field.lon.data}')
    
    #fig,ax=plt.subplots(1,1, constrained_layout=False, figsize=(24,6) )
    handle=ocean_section_plot(T0, ax, levels,cmap, refline=None, contour=contour,\
                                 xlimit=xl,\
                     colorbar=True, title=title, **kw)

    small_map(fig,ax,cent_lon=0,lat=None, lon=lon_sec)
    infofile = ('Sect'+ maintit + zlib.long_string(lon_sec)+ str(lmax) +'.pdf').replace(' ','_')
    print(f'PDF Figure Ready on file  --->  {infofile}')
    return infofile
@d.dedent
def lat_sect(fig, ax, field, lat_sec,labelR=[],view='Atlantic',
           maintit=[],lmax=1000, xl=None,cmap='coolwarm', contour=True,levels=[], sharp=[], **kw):

    '''
    Longitude section at a fixed latitude.

    This routine performs a section at the chosen latitude. The `sharpness` of he section is controlled by the parameter `sharp`. 
    A `sharp = 0` indicates a precise section at the nominal latitude, where any value different from zero will result
    in average on an interval of length 2*sharpness centered at the latitude.

    Parameters
    ==========
    
    fig:
        Figure to be used
    ax :
        Plot axis to be used
    field: xarray
        Data to be plotted
    lon_sec:
        Longitude of the section
    lmax:
        Maximum depth of the section
    view:
        * Atlantic      Atlantic at the center
        * Pacific       Pacific at the center
    xl:
        Extent of the section
    cmap:
        colormap
    contour:
        True/false for having contours
    levels:
        Contour Levels ( see `choose_contour`)
    sharp: positive float
        Semi-width of the averaging interval around the section.
        The average ignore NaN and therefore small features can be lost in the averaging.
    labelR:
        Label on the top right of the plot
    maintit:
        Main title
    
    Returns
    =======

    handle:
        handle to the plot
    '''

    print(f'Sharpness set to  {sharp}')

    #lon_sec = -140
    lev_min = 0
    lev_max = lmax
    #
    
    if lat_sec == 0:
        label2= zlib.lat_string(lat_sec) 
    else:
        label2= zlib.lat_string(lat_sec) + ' Latitude'
    title = {'maintitle':maintit,'lefttitle':label2,'righttitle':labelR}


    # Check sharpness
    if sharp > 0:
        T0 = field.sel(deptht=slice(lev_min,lev_max),lat=slice(lat_sec-sharp,lat_sec+sharp)).mean(dim='lat')
    elif sharp < 0:
        raise SystemExit(f'Wrong sharp in lat_sect --> {sharp}')
    else:
        try:
            T0 = field.sel(deptht=slice(lev_min,lev_max),lat=lat_sec)
        except:
            raise SystemExit(f'Latitude missing in data set  --> {field.lat.data}')
   
    if view == 'Atlantic':
        cent = 0.0
    else:
        cent=180.

    handle=ocean_section_plot(T0, ax, levels ,cmap, refline=None,contour=contour,\
                                 view=view,xlimit=xl,\
                     colorbar=True,title=title, **kw)
    small_map(fig,ax,cent_lon=cent,lat=lat_sec, lon=None)
   
    infofile = ('Sect'+ maintit + zlib.lat_string(lat_sec)+ str(lmax) + '.pdf').replace(' ','_')
    print(f'PDF Figure Ready on file  --->  {infofile}')
    return infofile
    
def changelabel(ax,fontsize=12,fontfamily='times',fontweight='normal'):
    '''
    Change label properties globally for all labels, for a single ax
    or all axes in figure

    Parameters
    ==========
    ax:
        axes of figure or single axes of panel
    
    fontsize:
        Fontsize

    fontfamily:
        FontFamily
    
    fontweight:
        FontWeight

    Examples
    ========

    >>> changelabel(ax,fontsize=12,fontfamily='times',fontweight='normal')
    '''
    if isinstance(ax, np.ndarray):
        panels = ax.flatten()
    else:
        panels=[ax]

    for i in panels:

        for label in i.get_xticklabels():
            label.set_fontfamily(fontfamily)
            label.set_fontsize(fontsize)
            label.set_fontweight(fontweight)
            #label.set_va('bottom')
            
        for label in i.get_yticklabels():
            label.set_fontfamily(fontfamily)
            label.set_fontsize(fontsize)
            label.set_fontweight(fontweight)
            label.set_ha('right')
        
    return

def changebox(ax,choice,linewidth=2,color='black',capstyle='round'):
    '''
    Change spines properties globally for all labels, for a single ax
    or all axes in figure.

    Parameters
    ==========
    ax:
        axes of figure or single axes of panel
    
    choice:
        The spines to be modified, either `all` or `top`,`right`,`left`,`bottom`
        If it is not `all` it could be a list..
        
    linewidth:
        linewidth
    
    color:
        color

    capstyle:


    Examples
    ========

    >>> changebox(ax,'all',linewidth=2,color='black',capstyle='round')
    >>> changebox(ax,['top','bottom'],linewidth=2,color='black',capstyle='round')
    >>> changebox(ax,'left',linewidth=2,color='black',capstyle='round')
    '''
    if isinstance(ax, np.ndarray):
        panels = ax.flatten()
    else:
        panels=[ax]

    if isinstance(choice,str):
        if choice == 'all':
            sides = ['top','bottom','left','right']
        else:
            sides = [choice]
    else:
        sides = choice

    for i in panels:
        for j in sides:
            i.spines[j].set_linewidth(linewidth)
            i.spines[j].set_color(color)
            i.spines[j].set_capstyle(capstyle)
        
    return

def label_equator(ax,short=True):
    '''
    Change latitudinal labels to 'EQ in the middle'

    '''

    if isinstance(ax, np.ndarray):
        panels = ax.flatten()
    else:
        panels=[ax]

    for i in panels:    
        i.xaxis.set_major_formatter(mticker.FuncFormatter(_update_ticks))
    
    return

def _update_ticks(x,pos):
    if x == 0:  
        return 'EQ'
    else:
        i, d = divmod(x, 1)
        if d > 0:
            if i < 0 :
                return str(-x) + 'S'
            else:
                return str(x) + 'N'
        else:
            if i < 0:
                return str(-int(i)) + 'S'
            else:
                return str(int(i)) + 'N'    
    return 

def zonal_streamfun_plot(datau,datav,ax,cont, color='black', refline=None,\
                     cmap='bwr', title={}, title_style={},label_style={},\
                     colorbar=True, smooth=True, special_value=9999):
    """
    Zonal  streamfunction.

    Plot meridional streamfunction for field datu e datav.

    Parameters      
    ----------
    datau : xarray
        X component of the streamlines   
    
    datav : xarray
        Y component of the streamlines
    
    ax :          
        Plot axis to be used        
    
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels  
            
    cmap:  
        Colormap
            
    smooth:   
        False/True if smoothing is desired
            
    colorbar:   
        False/True if a colorbar is desired      
    

    """
    
    #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }

   
    opt_label =  {**defaultGridLabels, **label_style}
    title_text =  {**defaultTitle, **title}
    opt_title =  {**defaultTitleStyle, **title_style}
   
    #Special Values
    datau=datau.where(datau != 9999.)
    datav=datav.where(datav != 9999.)

    #Eliminate extra dimensions
    if len(datau.shape) > 2:
        datau = datau.squeeze()
        datav = datav.squeeze()
    #Interpolate on a pressure regular grid
    U=datau.interp(pressure=np.arange(100,1000,50))
    V=datav.interp(pressure=np.arange(100,1000,50))
    #Compute Streamfunction

    X,Y = np.meshgrid(U.lat.data,U.pressure.data)
    # integrate
    intx=integrate.cumtrapz(V,X,axis=1,initial=0)[0]
    inty=integrate.cumtrapz(U,Y,axis=0,initial=0)

    psi1=-intx+inty

    intx2=integrate.cumtrapz(V,X,axis=1,initial=0)
    inty2=integrate.cumtrapz(U,Y,axis=0,initial=0)[:,0][:,None]

    psi2=-intx2+inty2

    #Xarray Data
    P =  xr.full_like(U,1.0)
    P.data =(psi1+psi2)/2.

    #Choose Contours
    vmin = P.min()
    vmax = P.max()
    cc = choose_contour(vmin,vmax,cont)
    print(f'Contouring from {vmin.data} to {vmax.data} with {cc} contours')
    
    
    # Contour Streamfunction
    
    handle = P.plot.contourf(
        ax=ax,                            # this is the axes we want to plot to
        cmap=cmap,                        # our special colormap
        levels=cc,      # contour levels specified outside this function
        xticks=np.arange(-90, 91, 15),  # nice x ticks
        yticks=[1000,850,700,500,300,200,100],    # nice y ticks
        add_colorbar=colorbar,               # don't add individual colorbars for each plot call
        add_labels=False                 # turn off xarray's automatic Lat, lon labels
    )
    if refline:
        hc = P.plot.contour(
        ax=ax,
        levels=refline,
        colors="k",  # note plurals in this and following kwargs
        linestyles="-",
        linewidths=1.25,
        add_labels=False  # again turn off automatic labels
        )
    
    #  Label the contours
    #     ax.clabel
    #         handles["contour"], fontsize=8, fmt="%.0f",  # Turn off decimal points
    #    )

    lev=U.pressure.values
    nlev=len(lev)
    ax.set_ylim(lev[nlev-1], lev[0])  # Invert y axis
    ax.set_xlim(90,-90)  # Invert x axis
    

    # Add gridlines
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_update_ticks))   
    set_titles_and_labels(ax,title_text,{'xlabel':'','ylabel':''},opt_title,opt_label)
   
    
    return handle

def add_cyclic_lon(data):
    '''
    Adding a cyclic longitude to an xarray
    (corrected error on data.lon[0])
    
    Parameters
    ---------
    data: xarray

    Returns
    -------
    xarray:
        Data with cyclic coordinate added
    '''

    
    nlat,nlon = data.shape
    newz = np.ones([nlat,nlon+1])
    dxlon = data.lon[1] -  data.lon[0]
    newlon = np.arange(data.lon[0],data.lon[-1]+2*dxlon,dxlon)
    newz[:,:-1] = data.data
    newz[:,-1] = newz[:,0]
    newZ = xr.DataArray(
                data   = newz,   # enter data here
                dims   = ['lat','lon'],
                coords = {'lat': data.lat,'lon':newlon})
    
    return newZ

def xstmap(U, V,  color='black', pro=None,ax=None, \
                      data_cent_lon=180, cyclic=True, coasts=True,\
                      lw=1,\
                      density=2,\
                      title=None, title_style=None,\
                      label_style=None,\
                      lonlabel=None,latlabel=None,\
                      xlimit=None,ylimit=None, \
                      colorbar=False,
                      cscale=None,cmap='coolwarm',norm=None, quiet=True):
    """
    Plot streamline for field U and V.

    This routine draws streamlines for the fields U,V, optionally colored with the field ``color``. Internally assume data is always on a latlon projection with central longitude at Greenwich.

    The data is supposed to be on a Plate (lat-lon) projection and the central longitude can be defined 
    via the paramater `data_cent_lon`. The defualt is `data_cent_lon=180`, meaning that the central longitude is over the pacific. 
    Coordinates that go from 0 to 360 implicitly assume such a Pacific central longitude.

    For the `Pacific` view it is assumed that the central longitude is at the dateline
    whereas for the `Atlantic` view the central longitude is at Greenwich.


    Parameters      
    ----------
    U : xarray
        X component of the streamlines    
    V : xarray
        Y component of the streamlines  
    color :   
        Color of the stremalines ('black'). If it is xarray color the streamlines with the colormap ``cmap``         
    cscale :
        Optionally, if an array is used in `color`, it normalize to cscale = ( min, max)
        If omitted, (min,max) is computed from the array        
    lw : float or array
        Linewidth, if a numpy array it is varying with the array values
    data_cent_lon : 
        Central longitude for the data projection
    latlabel:
        Position of Latitude grids
    lonlabel:
        Position of Longitude grids   
    density :    
        Density of the streamlines     
    ax :          
        Plot axis to be used            
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels      
    cmap :  
        Colormap     
    colorbar :   
        False/True   
    xlimit :       
        Limit of the map (lon)  
    ylimit :        
        Limit of the map (lat)  

    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """

    #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }
    defaultLatLabel = [-60.,-30.,0.,30.,60.]
    defaultLonLabel = np.arange(-180,181,30)

    if label_style is not None:
        opt_label =  label_style
    else:
        opt_label =  defaultGridLabels
    
    if title is not None:
        title_text = title
    else:
        title_text =  defaultTitle

    if title_style is not None:
        opt_title =  title_style
    else:
        opt_title =  defaultTitleStyle
    
    if latlabel is not None:
        opt_latlabel =  latlabel
    else:
        opt_latlabel =  defaultLatLabel
    
    if lonlabel is not None:
        opt_lonlabel =  lonlabel
    else:
        opt_lonlabel =  defaultLonLabel

    #Check right projection
    thispro = pro.__class__.__name__
    thisproview = pro.view
    # if not this  in  ['PlateCarree','NorthPolarStereo','SouthPolarStereo']:
    #     print(' Wrong Projection in `xmap` {}'.format(this))
    #     raise SystemExit
    
    #select color scale, if `color` is an array
    # colors are normalized accordin to `cscale`
    this = type(color).__name__
    print(f'Color scale chosen according to type {this}\n')
    normuv = None

    # Set data projection
    data_proj = car.PlateCarree(central_longitude=data_cent_lon)
    
    #Add Cyclic point
    if cyclic:
        U = add_cyclic_lon(U)
        V = add_cyclic_lon(V)

    #Adjust longitude for Atlantic projection
    if pro.view == 'Atlantic':
        U = adjust_data_centlon(U).assign_coords({'lon': (U.lon - 180)})
        V = adjust_data_centlon(V).assign_coords({'lon': (U.lon - 180)})


    #Eliminate extra dimensions
    if len(U.shape) > 2:
        U = U.squeeze()
        V = V.squeeze()
        
    # Add coastlines
    if coasts:
        ax.coastlines(linewidth=0.5)
    else:
        ax.coastlines(linewidths=0.5)
        ax.add_feature(cfeature.LAND, facecolor=color_land)

    #check colorscale
    if this == 'str':
        color_scale = color
    elif this == 'DataArray':
        if cyclic : color = add_cyclic_lon(color)
        if pro.view == 'Atlantic': 
            color=adjust_data_centlon(color).assign_coords({'lon': (color.lon - 180)})
        if not cscale:
            cscale= (color.min().min(), color.max().max() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        print(f'Color set an xarray -> cscale \n {cscale[0]} \t{cscale[1]}')
        color_scale = color.data      
    elif this == 'ndarray':
        if cyclic : color = add_cyclic_lon(color)  
        if not cscale:
            cscale= (color.max().max(), color.min().min() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        print(f'Color set an array -> cscale {cscale}')
        color_scale=color
    else:
        color_scale='black'
    
    if thispro  in  ['PlateCarree']:
        if ylimit is  None:
            ylim = ax.projection.y_limits
        else:
            ylim=ylimit
        ax.set_ylim(ylim)

        if xlimit is  None:
            xlim = ax.projection.x_limits
        else:
            xlim=xlimit
        ax.set_xlim(xlim)
    elif thispro in ['NorthPolarStereo','SouthPolarStereo']:
        if xlimit is  None:
            xlim = (np.amin(U.lon.values)-180,np.amax(U.lon.values)-180+0.001)
        else:
            xlim=xlimit
            
        if ylimit is  None:
            ylim = (np.amin(U.lat.values),np.amax(U.lat.values))
        else:
            ylim=ylimit
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )
        
        ax.set_extent(list(xlim+ylim),car.PlateCarree())  # 
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.spines['geo'].set_linewidth(2.0)
    else:
        print(f' Streamplot Projection in {thispro}')

    if not quiet:
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )

    handles = dict()
    # Stream-plot the data
    # There is no Xarray streamplot function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    handles = ax.streamplot(U.lon.data, U.lat.data, U.data, V.data, \
        linewidth=lw, density=density, color=color_scale, transform=data_proj,\
                      arrowstyle='fancy',cmap=cmap,  zorder=1, norm = normuv)
    
    
    # Add gridlines
    label_style = opt_label
    gl = ax.gridlines(draw_labels=True, dms=True,xlabel_style=label_style, ylabel_style=label_style, \
                  linewidth=0.6, color='gray', alpha=1.0,linestyle='--', \
                  formatter_kwargs={'degree_symbol':''})
    gl.right_labels = True
    gl.top_labels = False
    gl.xlocator = mticker.FixedLocator(opt_lonlabel)
    gl.ylocator = mticker.FixedLocator(opt_latlabel)
   
    set_titles_and_labels(ax,title_text,{'xlabel':'','ylabel':''},opt_title,opt_label)
    
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(handles.lines, cax=cax,orientation='vertical')
    
    return handles  

def vecmap(U, V,  C, color='black', pro=None,ax=None, \
                      data_cent_lon=0, cyclic=True, coasts=True,\
                      veckey=None,vec_min = 0.0, stride=10,\
                      title=None, title_style=None,\
                      label_style=None,\
                      lonlabel=None,latlabel=None,\
                      xlimit=None,ylimit=None, \
                      colorbar=False,
                      cscale=None,cmap='coolwarm',norm=None, quiet=True,**kw):
    """
    Vector plot for field U and V.

    This routine draws vectors for the fields U,V, optionally colored with the field ``color``. Internally assume data is always on a latlon projection with central longitude at Greenwich.

    Data must be on a (0,360) longitude coordinate referred to Greenwich, but the `xlimit` are
    referred to the projection chosen in `init`.

    For the `Pacific` view it is assumed that the central longitude is at the dateline
    whereas for the `Atlantic` view the central longitude is at Greenwich.

    It also takes all the other arguments for `quiver`

    Parameters      
    ----------
    U : xarray
        X component of the vectors    
    V : xarray
        Y component of the vectors  
    C : xarray
        Field to scale the vectors. If it is xarray color the streamlines with the colormap ``cmap``
    pro:
        Projection
    ax:
        ax to plot
    color :   
        Color of the stremalines ('black').          
    cscale :
        Optionally, if an array is used in `color`, it normalize to cscale = ( min, max)
        If omitted, (min,max) is computed from the array        
    data_cent_lon : 
        Central longitude for the data projection
    veckey : 
        It is a dictionary for the reference vector, the values are
            * X, Y -- Location in figure coordinates
            * U -- Reference Unit Length
            * label -- Reference string, it can be a latex string, i.e  r'$1 \\frac{m}{s}$'   
    vec_min :   
        Minimum value to represent with the vectors

    latlabel:
        Position of Latitude grids
    lonlabel:
        Position of Longitude grids             
    title : dict
        Dictionary with the title and its subtitles

        * lefttitle -- Title string on the left
        * righttitle -- Title string on the right
        * maintitle -- Title string at the center
    title_style : dict
        Dictionary with the style for the title
    label_style : dict
        Dictionary with the style for the labels      
    cmap :  
        Colormap     
    colorbar :   
        False/True   
    xlimit :       
        Limit of the map (lon)  
    ylimit :        
        Limit of the map (lat)  

    Returns
    -------

    handle :    
        Dictionary with matplotlib-like info on the plot
    
    """

    #Set default arguments
    defaultGridLabels = { "fontsize": 12, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitleStyle = { "fontsize": 14, "color": "black",'fontfamily':'futura','fontweight':'bold' }
    defaultTitle = { 'lefttitle':'','righttitle':'','maintitle':'' }
    defaultLatLabel = [-60.,-30.,0.,30.,60.]
    defaultLonLabel = np.arange(-180,181,30)

    if label_style is not None:
        opt_label =  label_style
    else:
        opt_label =  defaultGridLabels
    
    if title is not None:
        title_text = title
    else:
        title_text =  defaultTitle

    if title_style is not None:
        opt_title =  title_style
    else:
        opt_title =  defaultTitleStyle
    
    if latlabel is not None:
        opt_latlabel =  latlabel
    else:
        opt_latlabel =  defaultLatLabel
    
    if lonlabel is not None:
        opt_lonlabel =  lonlabel
    else:
        opt_lonlabel =  defaultLonLabel

    #Check right projection
    thispro = pro.__class__.__name__
    print(f'`Projection` chosen according to type {thispro}\n')
    
    normuv = None

    # Set data projection
    data_proj = car.PlateCarree(central_longitude=data_cent_lon)
    
    #Add Cyclic point
    if cyclic:
        U = add_cyclic_lon(U)
        V = add_cyclic_lon(V)
        C = add_cyclic_lon(C)

    #Eliminate extra dimensions
    if len(U.shape) > 2:
        U = U.squeeze()
        V = V.squeeze()
        C = C.squeeze()
    
        
    # Add coastlines
    if coasts:
        ax.coastlines(linewidth=0.5)
    else:
        ax.coastlines(linewidths=0.5)
        ax.add_feature(cfeature.LAND, facecolor=color_land)

    # Color array
    this = C.__class__.__name__
    if this == 'DataArray':
        print(f'Color set an xarray {C.name}')
        if not cscale:
            cscale= (C.min(), C.max() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        col_vec = C[::stride,::stride]      
    elif this == 'ndarray':
        print(f'Color set an array {C.name}')
        if not cscale:
            cscale= (C.max(), C.min() )
        normuv = plt.cm.colors.Normalize(cscale[0],cscale[1])
        col_vec=C[::stride,::stride]
    else:
        col_vec = None

    
    if thispro  in  ['PlateCarree']:
        if ylimit is  None:
            ylim = ax.projection.y_limits
        else:
            ylim=ylimit
        ax.set_ylim(ylim)

        if xlimit is  None:
            xlim = ax.projection.x_limits
        else:
            xlim=xlimit
        ax.set_xlim(xlim)
    elif thispro in ['NorthPolarStereo','SouthPolarStereo']:
        if xlimit is  None:
            xlim = (np.amin(U.lon.values)-180,np.amax(U.lon.values)-180+0.001)
        else:
            xlim=xlimit
            
        if ylimit is  None:
            ylim = (np.amin(U.lat.values),np.amax(U.lat.values))
        else:
            ylim=ylimit
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )
        
        ax.set_extent(list(xlim+ylim),car.PlateCarree())  # 
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.spines['geo'].set_linewidth(2.0)
    else:
        print(f' Projection chosen {thispro}')

    if not quiet:
        print(' Plotting with x limits {}  '.format(xlim)  ) 
        print(' Plotting with y limits {}  '.format(ylim) )

    if not quiet:
        print(f'Vecmap  \n')

     #Eliminate small values
    mag = np.sqrt(U*U + V*V)
    this_U = U.where(mag > vec_min)
    this_V = V.where(mag > vec_min)

    # Vector the data
    # There is no Xarray quiver function, yet. So need to call matplotlib.streamplot directly. Not sure why, but can't
    # pass xarray.DataArray objects directly: fetch NumPy arrays via 'data' attribute'
    # sp=ax.quiver(this_U.lon.data[::stride], this_U.lat.data[::stride], this_U.data[::stride,::stride], this_V.data[::stride,::stride], col_vec,\
    #                     linewidth=1, color=color, \
    #                     cmap=cmap,  zorder=1,norm = normuv,**kw)
    sp=ax.quiver(this_U.lon.data[::stride], this_U.lat.data[::stride], this_U.data[::stride,::stride], this_V.data[::stride,::stride], col_vec,\
                         transform=data_proj,color=color, cmap=cmap,  norm=normuv, zorder=1,**kw)

    if veckey:
        ax.quiverkey( sp, veckey['X'], veckey['Y'], veckey['Unit'], veckey['label'],coordinates='figure')

    #
    
    # Add gridlines
    label_style = opt_label
    gl = ax.gridlines(draw_labels=True, dms=True,xlabel_style=label_style, ylabel_style=label_style, \
                  linewidth=0.6, color='gray', alpha=1.0,linestyle='--', \
                  formatter_kwargs={'degree_symbol':''})
    gl.right_labels = True
    gl.top_labels = False
    gl.xlocator = mticker.FixedLocator(opt_lonlabel)
    gl.ylocator = mticker.FixedLocator(opt_latlabel)
   
    set_titles_and_labels(ax,title_text,{'xlabel':'','ylabel':''},opt_title,opt_label)
   
    # Add colorbar
    if colorbar:
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = tl.make_axes_locatable(ax)
        cax = divider.append_axes('right',size="2.5%", pad=0.2, axes_class=plt.Axes)
        ax.get_figure().colorbar(sp, cax=cax,orientation='vertical')
    
    return {'Vectors':sp,'Gridlines':gl} 