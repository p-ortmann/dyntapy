# Dyntapy - Static and Dynamic Macroscopic Traffic Assignment in Python
[![pipeline status](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/badges/master/pipeline.svg)](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/-/commits/master)
[![Documentation Status](https://readthedocs.org/projects/dyntapy/badge/?version=latest)](https://dyntapy.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![conda verson](https://anaconda.org/conda-forge/dyntapy/badges/version.svg)](https://anaconda.org/conda-forge/dyntapy)
[![status](https://joss.theoj.org/papers/e35f54090c93bebeba060a8c19ae3d89/status.svg)](https://joss.theoj.org/papers/e35f54090c93bebeba060a8c19ae3d89)

dyntapy is a macroscopic car traffic modelling toolkit that offers both Static and Dynamic
assignment algorithms and support for visualizing assignment results in 
jupyter notebooks.
It is designed to be extendable. Other assignment algorithms can be added rather easily, 
building on the provided definitions of supply, demand and assignment results.

Provided functionalities:
- Network generation from OSM using OSMnx complying with [GMNS](https://github.com/zephyr-data-specs/GMNS) attribute names.
- Static Assignments (deterministic user equilibrium: DialB, MSA; stochastic, uncongested: Dial's Algorithm)
- Dynamic User Equilibrium using the iterative link transmission model
- Visualization of static and dynamic attributes, including bidirectional flow visualization
- Fast shortest path computations
- Selected Link Analysis


There are demo tutorials available that you can run in Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy-tutorials/HEAD)

These tutorials are running with the latest released version of dyntapy. 

For the development version based on this repository, go [here](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy/HEAD?urlpath=/tree/tutorials).

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
Using -e makes the repository editable.

# Visualization of Networks and Assignment Results

We briefly demonstrate here the visualization capabilities within dyntapy, for a more detailed look
at all the provided functionality we refer to the tutorials.

## Static Assignment Results

On the left: Results of an assignment with Dial's Algorithm B. \
On the right: Selected Link Analysis of the link highlighted in pink. 

The color transition from green to red mirror the volume-to-capacity ratio from 0 to >1.
The width of the links is scaled to the largest flow in the network.


<img src="./tutorials/imgs/assignment_dial_b.png"  width="500" height="500">
<img src="./tutorials/imgs/selected_link_analysis.png"  width="500" height="500">

## Dynamic Assignment Results

For analysis of dynamic scenarios a time slider can be used to step through different time discretization steps.
The state of the plot is updated as the time slider is moved, the data that can be attained by hovering over elements is also changed dynamically.

<img src="./tutorials/imgs/visualizing_dta.gif"  width="60%" height="60%">


# Related Python Packages
[AequilibraE](http://aequilibrae.com/python/latest/)
is a traffic modelling kit that supports a set of static assignments paired with OpenStreetMap parsing.
It aims to be an open source alternative to Visum
and provides comprehensive visualization and network management capabilities in QGIS.
dyntapy differs from AequilibraE mainly by its inclusion of dynamic models and command-driven 
visualization capabilities within jupyter notebooks.

Similar to other popular graph packages, such as [igraph](https://igraph.org/) and [networkx](https://networkx.org/), 
we offer shortest path computations. Our implementations are more than twice as fast as igraph
and more than 20x faster than networkx for road networks.
# Acknowledgements

 A part of the development of this package has been funded by DUET - [Digital Urban European Twins](https://www.digitalurbantwins.com/) as part of the Horizon 2020 funding program (Grant ID 870607).  
