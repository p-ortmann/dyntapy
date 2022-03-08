# Dyntapy - Static and Dynamic Macroscopic Traffic Assignment in Python
[![pipeline status](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/badges/master/pipeline.svg)](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/-/commits/master)
[![Documentation Status](https://readthedocs.org/projects/dyntapy/badge/?version=latest)](https://dyntapy.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![conda verson](https://anaconda.org/conda-forge/dyntapy/badges/version.svg)](https://anaconda.org/conda-forge/dyntapy)




Provided functionalities:
- Network generation from OSM using OSMnx complying with [GMNS](https://github.com/zephyr-data-specs/GMNS) attribute names.
- Static Assignments (deterministic user equilibrium: DialB, MSA; stochastic, uncongested: Dial's Algorithm)
- Dynamic User Equilibrium using the iterative link transmission model [^1]
- Visualization of static and dynamic attributes, including bidirectional flow visualization
- fast shortest path computations
- Selected Link Analysis


[^1]: convergence below $`10^{-2}`$ according to the provided excess cost criteria cannot be guaranteed for a reasonable amount of iterations

There are demo tutorials available that you can run in Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy/HEAD?urlpath=/tree/tutorials)

# How to install


dyntapy is available on conda-forge.
Using
```shell
conda install -c conda-forge dyntapy
```
will install dyntapy in your current environment.

Note that dyntapy is also available on pip, however we don't recommend installing
it this way since some geo-dependencies, namely shapely, pyproj and fiona, rely on 
C-based libraries that cannot be installed via pip.

## from this repository 

This is only for those who want to have the latest state of development of dyntapy.

Download or clone the repository. We assume you have conda installed.
Navigate to the folder containing the environment.yml or pass on the full path.
```shell
conda env create environment.yml 
```
This will sort out dyntapy's dependencies using conda.

we now can install the package with
```shell
python -m pip install -e path-to-folder
```
Using -e makes the repo editable.
If you make changes or add a functionality it will be available in a fresh session
or if you reload the module.
