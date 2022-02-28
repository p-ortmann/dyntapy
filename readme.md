# Dyntapy - Dynamic Traffic Assignment in Python
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy/HEAD?urlpath=/tree/tutorials)
[![pipeline status](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/badges/master/pipeline.svg)](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/-/commits/master)


Provided functionalities:
- Network generation from OSM using OSMnx complying with [GMNS](https://github.com/zephyr-data-specs/GMNS) attribute names.
- Static Assignments (deterministic user equilibrium: DialB, MSA; stochastic, uncongested: Dial's Algorithm)
- Dynamic User Equilibrium using the iterative link transmission model [^1]
- Visualization of static and dynamic attributes, including bidirectional flow visualization
- fast shortest path computations
- Selected Link Analysis


[^1]: convergence below $`10^{-2}`$ according to the provided excess cost criteria cannot be guaranteed for a reasonable amount of iterations

There are demo tutorials available that you can run in Binder.

# How to install
Download the provided environment.yml. We assume you have conda installed. 
navigate to the folder containing the environment.yml or pass on the full path.
```shell
conda env create environment.yml 
```
This will sort out dyntapy's dependencies using conda and install dyntapy itself via pip.

## from this repository 
Download the repository, follow the steps above to get all of dyntapy's dependencies.
we now can install the package with
```shell
python -m pip install -e path-to-folder
```
Using -e makes the repo editable.
If you make changes or add a functionality it will be available in a fresh session
or if you reload the module.
verify that importing works as expected, open the interpreter
```shell
python
```
and try
```python
import dyntapy
```
voila!
