# Dyntapy - Dynamic Traffic Assignment in Python
Provided functionalities:
- Network generation from OSM using OSMnx complying with [GMNS](https://github.com/zephyr-data-specs/GMNS) attribute names.
- Static Assignments (deterministic user equilibrium: FW, DialB, MSA; stochastic, uncongested: Dial's Algorithm)
- Dynamic User Equilibrium using the iterative link transmission model [^1]
- Visualization of real and toy networks with Static and Dynamic attributes using Bokeh, including bidirectional flow visualization

[^1]: convergence according to the provided excess cost criteria cannot be guaranteed for a reasonable amount of iterations

There are demo tutorials available that you can run in Binder.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy/HEAD)

# How to install
If you want this to be part of a particular conda environment, activate it first.
```shell
conda activate your-environment
```
## from PyPi
dyntapy is available from PyPi 
```shell
python -m pip install dyntapy
```
## from this repository 
Download the repository
we now can install the package with
```shell
python -m pip install -e path-to-folder
```
pip automatically pulls all the dependencies that are listed in the setup.py.
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
