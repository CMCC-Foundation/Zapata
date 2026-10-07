"""
Zapata
------

Zapata is a small Python package used in this project. Current modules and
subpackages present in this directory:

- lib.py
  - general-purpose helper functions and utilities used across the package.
- computation.py
  - numerical and domain-specific computation routines.
- mapping.py
  - spatial/geographical mapping helpers and utilities.
- data.py
  - data loading, parsing and lightweight data models / I/O helpers.
- SciVis_colormaps/
  - directory containing colormap definitions and visualization palettes.
- __pycache__/, .DS_Store.gz
  - runtime and OS artifacts (ignored by version control).

Notes
- Keep this docstring synchronized with the repository when files or folders
  are added or removed.
- Consider re-exporting commonly used symbols from submodules here for
  convenience (e.g., from .data import load_data).
"""

__version__ = "1.0.0"
__author__ = "Antonio Navarra"


