# import ipywidgets as widgets
import os
from IPython.display import Javascript

from typing import Sequence, Mapping, Iterable, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


import numpy as np
import scipy.linalg as sc
import pandas as pd
import xarray as xr

import sys
import platform
import pkg_resources

if os.environ.get("CONDA_DEFAULT_ENV") == 'BOOKOORT' or os.environ.get("CONDA_DEFAULT_ENV") == 'ERA5':   
    import ddsapi
else:
    print('ddsapi not available, not in BOOKOORT or ERA5 conda environment')

def name_notebook(newname):
    '''
    Change name to Jupyterlab instance
    '''
    Javascript('document.title="{}"'.format(newname))
def get_values_from_dict(input_dict, keys):
    '''
    Get values from dictionary `input_dict` for keys in `keys`

    Parameters
    ==========
    input_dict: 
        Dictionary  
    keys:
        List of keys
    
    Returns
    =======
    List of values
    
    '''
    return [input_dict[key] for key in keys if key in input_dict]

    
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

def get_environment_info(option):
    '''
    Get information about the Python environment

    Parameters
    ==========

    option:
        String
        Options are:
        'interpreter': Get the path of the Python interpreter
        'version': Get the Python version
        'packages': Get the list of installed packages

    Returns
    =======

    Information about the Python environment
    '''
    python_executable = sys.executable
    python_version = platform.python_version()
    installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
    match option:
        case 'interpreter':
            return python_executable
        case 'version':
            return python_version
        case 'packages':
            return installed_packages
        case 'all':
           return python_executable, python_version, installed_packages
        case _:
            print('Choose an option: interpreter, version, packages, all')
    return 



# ─── a single executor shared by all calls ────────────────────────────────────
_dl_pool = ThreadPoolExecutor(max_workers=4)                    # tweak as needed

# Pre-compute the static list so we don’t re-allocate it every call
_VERTICAL_LEVELS = [
    0.5057600140571594, 1.5558552742004395, 2.6676816940307617, 3.8562798500061035,
    5.140361309051514, 6.543033599853516, 8.09251880645752, 9.822750091552734,
    11.773679733276367, 13.99103832244873, 16.52532196044922, 19.42980194091797,
    22.75761604309082, 26.558300018310547, 30.874561309814453, 35.740203857421875,
    41.180023193359375, 47.21189498901367, 53.85063552856445, 61.11283874511719,
    69.02168273925781, 77.61116027832031, 86.92942810058594, 97.04131317138672,
    108.03028106689453, 120, 133.07582092285156, 147.40625, 163.16445922851562,
    180.5499267578125, 199.7899627685547, 221.14117431640625, 244.890625,
    271.35638427734375, 300.88751220703125, 333.8628234863281, 370.6884765625,
    411.7938537597656, 457.6256103515625, 508.639892578125, 565.2922973632812,
    628.0260009765625, 697.2586669921875, 773.3682861328125, 856.678955078125,
    947.4478759765625, 1045.854248046875, 1151.9912109375, 1265.8614501953125,
    1387.376953125, 1516.3636474609375, 1652.5684814453125, 1795.6707763671875,
    1945.2955322265625, 2101.026611328125, 2262.421630859375, 2429.025146484375,
    2600.38037109375, 2776.039306640625, 2955.5703125, 3138.56494140625,
    3324.640869140625, 3513.445556640625, 3704.65673828125, 3897.98193359375,
    4093.15869140625, 4289.95263671875, 4488.15478515625, 4687.5810546875,
    4888.06982421875, 5089.478515625, 5291.68310546875, 5494.5751953125,
    5698.060546875, 5902.0576171875
]

def get_ocean_GLORS(year: int | str, var: str, outfile: str, *, pool: ThreadPoolExecutor = _dl_pool):
    """
    Asynchronously download GLORS reanalysis for `year` into `outfile`.

    Parameters
    ----------
    year : int or str
        Year to fetch (e.g. 1993).  Converted to str internally.
    var : str
        Variable to fetch, e.g. 'sea_water_potential_temperature'.
    outfile : str
        Path to write the NetCDF file.
    pool : ThreadPoolExecutor, optional
        Executor that runs the blocking retrieval.  Defaults to a module-level
        pool so every call shares the same threads.

    Returns
    -------
    concurrent.futures.Future
        Future whose `.result()` blocks until the file is fully written.
    """

    # Ensure `year` is a string for the DDS request
    year_str = str(year)

    # Function that does the real work (runs in a background thread)
    def _download():
        client = ddsapi.Client()
        client.retrieve(
            "cglorsv7",
            "monthly-regular",
            {
                "vertical" : _VERTICAL_LEVELS,
                "variable" : var,
                "time"     : {"year": [year_str],
                              "month": [str(m) for m in range(1, 13)]},
                "format"   : "netcdf",
            },
            outfile,
        )
        print(f"Data for {year_str} saved to {outfile}")

    # Submit the job to the shared thread pool and return the Future
    return pool.submit(_download)

    
def write_netcdf(ds,file):
    """
    Write a NetCDF file with the given filename.
    
    Parameters
    ==========
    file: 
        Name of the NetCDF file to be written.
    
    Returns
    =======
    None
    """
    # Define “maximum” loss-less compression settings
    compression_settings = {
        # Either syntax works; pick the one your h5netcdf/xarray combo supports
        # 1) NetCDF-style keywords (widely supported)
        "zlib": True,
        "complevel": 9,
        # 2) Direct HDF5 keywords (h5netcdf ≥ 1.0)
        # "compression": "gzip",
        # "compression_opts": 9,
    }

    # Apply the same settings to every variable (data + coords if you like)
    encoding = {var: compression_settings for var in ds.variables}

    # Write the file
    ds.to_netcdf(
        file,          # destination
        mode='w',  # write mode
        engine="h5netcdf",    # use the pure-Python HDF5 backend
        encoding=encoding,    # per-variable compression map
    )

    return



from functools import partial

# import xarray as xr
# from functools import partial
# from typing import Sequence, Mapping, Iterable, Callable, Optional, Dict

# ──────────────────────────────────────────────────────────────────────────────
def _month_filter(
    ds: xr.Dataset,
    *,
    months: Iterable[int],
    time_dim: str,
    weighted: bool,
    avg_per_file: bool,
    user_pp: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Internal helper used as `preprocess` by open_mfdataset.

    1. runs any user-supplied preprocessing,
    2. keeps only the requested months,
    3. *optionally* averages those months inside the single file
       (weighted by month length if requested).
    """
    if user_pp is not None:
        ds = user_pp(ds)

    # ------------- 2. sub-select months --------------------------------------
    ds = ds.sel({time_dim: ds[time_dim].dt.month.isin(months)})

    # ------------- 3. optionally aggregate *inside* each file ---------------
    if avg_per_file and ds[time_dim].size:
        if weighted:
            w = ds[time_dim].dt.days_in_month
            ds = (ds * w).sum(time_dim) / w.sum(time_dim)
        else:
            ds = ds.mean(time_dim)

        # keep a stub time coordinate (first month) so concat works cleanly
        first_stamp = ds[time_dim][0].values if time_dim in ds.coords else None
        ds = ds.expand_dims({time_dim: [first_stamp]})

    return ds
# ──────────────────────────────────────────────────────────────────────────────

def _month_filter(
    ds: xr.Dataset,
    months: Iterable[int],
    time_dim: str,
    weighted: bool,
    avg_per_file: bool,
    user_pp: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset:
    if user_pp is not None:
        ds = user_pp(ds)

    ds = ds.sel({time_dim: ds[time_dim].dt.month.isin(months)})

    if avg_per_file:
        if ds[time_dim].size == 0:
            return ds.drop_vars(time_dim, errors="ignore")
        if weighted:
            w = ds[time_dim].dt.days_in_month
            return (ds * w).sum(time_dim) / w.sum(time_dim)
        else:
            return ds.mean(time_dim)

    return ds

def build_preprocess(months, time_dim, weighted, avg_per_file, user_pp):
    def _pre(ds):
        return _month_filter(
            ds,
            months=months,
            time_dim=time_dim,
            weighted=weighted,
            avg_per_file=avg_per_file,
            user_pp=user_pp,
        )
    return _pre

def seasonal_average(
    files: Sequence[str],
    seasons: Mapping[str, Iterable[int]],
    *,
    time_dim: str = "time",
    weighted: bool = True,
    avg_per_file: bool = False,
    user_preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
    **open_kwargs,
) -> Dict[str, xr.Dataset]:
    """
    Compute seasonal averages from a collection of NetCDF files using xarray.

    This function loads multiple files, selects data for specified seasons (e.g., DJF, JJA),
    and computes weighted or unweighted averages over the time dimension. Handles special
    cases like DJF (December-January-February) that span calendar years. Optionally averages
    per file or after merging all files. Ensures all file handles are closed after processing.

    Uses xarray's open_mfdataset, which employs a lazy loading mechanism: data is not read
    into memory immediately, but rather loaded on demand as computations are performed. This
    allows efficient handling of large datasets and parallel processing, but requires explicit
    closing of datasets to release file handles and resources. All file handles are closed
    automatically in this function after processing each season.

    Parameters
    ----------
    files : Sequence[str]
        List of file paths to NetCDF files to process.
    seasons : Mapping[str, Iterable[int]]
        Dictionary mapping season names (e.g., 'DJF', 'JJA') to lists of month numbers.
        Example: {'DJF': [12, 1, 2], 'JJA': [6, 7, 8]}
    time_dim : str, default 'time'
        Name of the time dimension in the dataset.
    weighted : bool, default True
        If True, compute weighted averages using days in month; otherwise, use simple mean.
    avg_per_file : bool, default False
        If True, average each file separately before combining; if False, merge then average.
        For DJF, this is forced to False to preserve continuity across years.
    user_preprocess : Callable[[xr.Dataset], xr.Dataset] or None, default None
        Optional user-supplied preprocessing function applied to each dataset before averaging.
    **open_kwargs
        Additional keyword arguments passed to xarray.open_mfdataset.

    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary mapping season names to xarray Datasets containing the seasonal averages.
        Each Dataset includes metadata about the averaging method and selected months.

    Notes
    -----
    - Uses xarray.open_mfdataset for efficient multi-file loading.
    - DJF is handled specially to ensure correct grouping across years.
    - All datasets are closed after processing to avoid file handle leaks.
    - Raises ValueError if no data is found for a season or if DJF grouping is incomplete.

    Example
    -------
    >>> files = ["data1.nc", "data2.nc"]
    >>> seasons = {"DJF": [12, 1, 2], "JJA": [6, 7, 8]}
    >>> result = seasonal_average(files, seasons)
    >>> djf_avg = result["DJF"]
    """
    out: Dict[str, xr.Dataset] = {}

    for name, months in seasons.items():
        print(f"Processing season: {name} with months {months}...")
        is_djf = set(months) == {12, 1, 2}

        if is_djf and avg_per_file:
            print("  NOTE: DJF spans calendar years and cannot be averaged per file correctly.")
            print("  Forcing avg_per_file=False to preserve seasonal continuity across years.")
            avg_per_file = False

        preprocess = build_preprocess(
            months=months,
            time_dim=time_dim,
            weighted=weighted,
            avg_per_file=avg_per_file,
            user_pp=user_preprocess,
        )

        local_open_kwargs = open_kwargs.copy()
        if avg_per_file:
            local_open_kwargs.update({"combine": "nested", "concat_dim": "file"})
        else:
            local_open_kwargs.setdefault("engine", "h5netcdf")
            local_open_kwargs.setdefault("lock", False)
            local_open_kwargs.setdefault("parallel", False)

        print("  Opening multiple files with open_mfdataset...")
        ds = xr.open_mfdataset(files, preprocess=preprocess, **local_open_kwargs)
        try:
            if not avg_per_file:
                if time_dim not in ds.dims or ds[time_dim].size == 0:
                    raise ValueError(f"No data for season '{name}' in the provided files.")

            if avg_per_file:
                print("  Averaging across files...")
                out[name] = ds.mean("file")
            else:
                print("  Applying seasonal average logic...")
                time_vals = pd.to_datetime(ds[time_dim].values)
                ds = ds.assign_coords({time_dim: time_vals})

                if is_djf:
                    ds = ds.sel({time_dim: ds[time_dim].dt.month.isin([12, 1, 2])})
                    months = ds[time_dim].dt.month
                    years = ds[time_dim].dt.year.where(months != 12, ds[time_dim].dt.year + 1)
                    ds_grouped = ds.groupby(years.rename("djf_year"))

                    valid_groups = []
                    for year, group in ds_grouped:
                        if group[time_dim].size == 3:
                            if weighted:
                                w = group[time_dim].dt.days_in_month
                                weighted_avg = (group * w).sum(time_dim) / w.sum(time_dim)
                            else:
                                weighted_avg = group.mean(time_dim)
                            valid_groups.append(weighted_avg)

                    if not valid_groups:
                        raise ValueError("No complete DJF season found.")

                    out[name] = xr.concat(valid_groups, dim="djf_year").mean("djf_year")
                else:
                    if weighted:
                        w = ds[time_dim].dt.days_in_month
                        out[name] = (ds * w).sum(time_dim) / w.sum(time_dim)
                    else:
                        out[name] = ds.mean(time_dim)

            out[name].attrs.update(ds.attrs)
            out[name].attrs["season_months"] = list(months)
            out[name].attrs["season_weighted"] = "Weighted" if weighted else "Unweighted"
            out[name].attrs["season_avg_per_file"] = (
                "Average per file" if avg_per_file else "Average after merging"
            )
            out[name].attrs["season_name"] = name
            out[name].attrs["season_time_dim"] = time_dim

            print(f"  Finished processing season: {name}\n")
        finally:
            ds.close()

    return out





def select_files_by_years(YEARS, file_template="GLORS_YEARS_T_44_{year}.nc", directory="."):
    """
    Selects files based on the years provided, checks their existence in the specified directory,
    and returns a list of selected file paths along with a sorted list of years for which files were found.
    
    Parameters
    ==========

    YEARS (list): List of years to check.
    file_template (str): Template for the file name, e.g., "GLORS_YEARS_T_44_{year}.nc".
    directory (str): Directory path where the files are stored.

    Returns
    =======


    tuple: A tuple containing:
        - selected_files (list): List of file paths found.
        - sorted_found_years (list): Sorted list of years for which the files were found.
    """
    selected_files = []
    found_years = []  # To store years for which the file was found
    
    for year in YEARS:
        # Construct the filename based on the current year
        filename = file_template.format(year=year)
        
        # Check if the file exists in the directory
        file_path = os.path.join(directory, filename)
        print(f"Checking for file: {file_path}")
        if os.path.exists(file_path):
            selected_files.append(file_path)
            found_years.append(year)  # Add the year to the found_years list
        else:
            print(f"File not found: {filename}")
    
    # Sort the list of found years
    sorted_found_years = sorted(found_years)

    return sorted(selected_files), sorted_found_years


# read a text (txt) file and return an numpy array

def read_txt_file(file_path: str) -> np.ndarray:
    """
    Read a text file and return its content as a numpy array.
    Parameters
    ==========
    file_path:
        Path to the text file to be read.
    Returns
    =======
    numpy.ndarray:
        Numpy array containing the data from the text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, "r") as f:
        data = f.readlines()
    return np.array([line.strip().split() for line in data],dtype='float')
#Write a numpy array to a text file
def write_txt_file(data: np.ndarray, file_path: str) -> None:
    """
    Write a numpy array to a text file.
    
    Parameters
    ==========
    data:
        Numpy array to be written to the file.
    file_path:
        Path where the text file will be saved.
    
    Returns
    =======
    None
    """
    np.savetxt(file_path, data, fmt='%s')
    return

def optimize_for_macos():
    """Apply macOS-specific optimizations."""
    # Use all available cores efficiently
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
    
    # Optimize numpy for Apple Silicon if available
    if sys.platform == "darwin" and hasattr(np, "__config__"):
        # Enable accelerated BLAS on macOS
        os.environ["OPENBLAS_NUM_THREADS"] = str(mp.cpu_count())
    
    # Set xarray/dask to use multiple threads
    import dask
    dask.config.set(scheduler='threads')
    dask.config.set(num_workers=mp.cpu_count())
