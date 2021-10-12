## How to install

we assume that you already have a virtual environment set up with conda.
1. open a command line tool, on windows hit start and look for anaconda prompt.
 - clone this repository with
```shell
git clone git@gitlab.kuleuven.be:ITSCreaLab/mobilitytoolsresearch/dyntapy.git path-to-your-folder
```
 you may first have to set up ssh, or use https instead:
```shell
git clone https://gitlab.kuleuven.be/ITSCreaLab/mobilitytoolsresearch/dyntapy.git path-to-your-folder
```
see here https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html for some background.
You may also download it, make sure to extract before proceeding with the next step.
2. assuming you use conda, next you have to activate the environment you want to use 
```shell
conda activate myenv
```
we now can install the package with
```shell
python -m pip install -e path-to-your-folder 
```
pip automatically pulls all the dependencies that are listed in the requirements.txt, see setup.py.
Using -e makes the repo editable. 
If you make changes or add a functionality it will be available in a fresh session 
or if you reload the module.
3. verify that importing works as expected, open the interpreter
```shell
python
```
and try
```python
import dyntapy
```
voila!
Demos of the provided functionality can be found in the notebooks under >>tutorials<<, some more examples can be 
found under >>testing<<
you can experiment with the notebooks in binder without the need to install anything.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.kuleuven.be%2FITSCreaLab%2Fpublic-toolboxes%2Fdyntapy/HEAD)
