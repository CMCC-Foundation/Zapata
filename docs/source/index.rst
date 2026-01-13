Welcome to CMCC Data Analysis Library !
=======================================

This is the assembled library of codes jointly developed at CMCC.
It contains mapping, interpolation and data intake functions for working with CMCC Supercomputing Center,
accessing to datasets on remote storage systems and Zeus cluster computational resources.

Access credentials for CMCC SCC facility can be obtained upon request to hsm@cmcc.it.

The source of this library is available at the `Download` links on the left.

The library is best used together with the Anaconda distribution. The environment must be traced
from the individual packages, but standard packages like `numpy`, `scipy`, `matplotlib` are definitely involved.
Zapata relies also heavily on `xarray` and `netcdf` with the HDR5 version.

Zapata is also best used in conjunction with Jupyterlab Notebooks.



.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:
   
   interp
   klus
   zapata
   datainfo
   Download
   conda_reference
   Release_Notes
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
