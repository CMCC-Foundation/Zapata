=====================
Conda useful commands
=====================

Getting Started
---------------

.. csv-table:: 
    :header: "Task", "Conda Command"
    :widths: 40, 30

    "Verify Conda is installed, check version number", "conda info"
    "Update Conda to the current version", "conda update -n base conda" 
    "Update all packages to the latest version of Anaconda. Will install stable and compatible versions, 
    not necessarily the very latest.", "conda update anaconda"

Working with Environments
-------------------------

.. csv-table:: 
    :header: "Task", "Conda Command"
    :widths: 40, 30
    :escape: \

    "Create a new environment named ENVNAME with specific version of Python and packages installed."  , "conda create --name ENVNAME python=3.6 \"PKG1>7.6\" PKG2"
    "Activate a named Conda environment", "conda activate ENVNAME"
    "Deactivate current environment", "conda deactivate"
    "List all packages and versions in the active environment", "conda list"
    "List all revisions made within the active environment", "conda list --revisions"
    "List all packages and versions in a named environment", "conda list --name ENVNAME "
    "List all revisions made in a specified environment","conda list --name ENVNAME --revisions"
    "Restore an environment to a previous revision", "conda install --name ENVNAME --revision REV_NUMBER"
    "Delete an entire environment","conda remove -name ENVNAME --all"
    "Install packages from a file into environment", "conda install --name ENVNAME --file"
    
Sharing Environments
--------------------

.. csv-table:: 
    :header: "Task", "Conda Command"
    :widths: 40, 30

    "Make an exact copy of an environment", "conda create --clone ENVNAME --name NEWENV"
    "Export an environment to a YAML file that can be read on Windows, macOS, and Linux", "conda env export --name ENVNAME > envname.yml"
    "Create an environment from YAML file", "conda env create --file envname.yml"
    "Create an environment from the file named environment.yml in the current directory","conda env create"
    "Export an environment with exact package versions for one OS", "conda list --explicit > pkgs.txt"
    "Create an environment based on exact package versions", "conda create --name NEWENV --file pkgs.txt"

Using Packages and Channels
---------------------------

.. csv-table:: 
    :header: "Task", "Conda Command"
    :widths: 40, 30
    :escape: \

    "Find a package on all channels using the Anaconda Client", "anaconda search FUZZYNAME"
    "Install package from a specific channel", "conda install conda-forge::PKGNAME"
    "Install a package by exact version number (3.1.4)","conda install PKGNAME==3.1.4"
    "Install one of the listed versions (OR)","conda install \"PKGNAME[version='3.1.2|3.1.4']\" "
    "Install following several constraints (AND)","conda install \"PKGNAME>2.5,<3.2\" "
    "Add a channel to your Conda configuration","conda config --add channels CHANNELNAME"


Additional Useful Hints
-----------------------

.. csv-table:: 
    :header: "Task", "Conda Command"
    :widths: 40, 30

    "Detailed information about package versions", "conda search PKGNAME --info"
    "Remove unused cached files including unused packages", "conda clean --all"
    "Remove a package from an environment", "conda uninstall PKGNAME --name ENVNAME"
    "Update all packages within an environment", "conda update --all --name ENVNAME"
    "Run most commands without requiring a user prompt. Useful for scripts.", "conda install --yes PKG1 PKG2"



 
