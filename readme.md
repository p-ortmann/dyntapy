###How to install

we assume that you already have a virtual environment set up with conda.
1.
open a command line tool, on windows hit start and look for anaconda prompt.
 - clone this repository with
```shell
git clone git@gitlab.kuleuven.be:ITSCreaLab/mobilitytoolsresearch/dtapy.git path-to-your-folder
```
 you may first have to set up ssh, or use https instead:
```shell
git clone https://gitlab.kuleuven.be/ITSCreaLab/mobilitytoolsresearch/dtapy.git path-to-your-folder
```
see here https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html for some background.
You may also download it, make sure to extract before proceeding with the next step.
2. 
assuming you use conda, next you have to activate the environment you want to use 
```shell
conda activate myenv
```
we now can install the package with
```shell
python -m pip install -e path-to-your-folder 
```
3.
verify that importing works as expected, open the interpreter
```shell
python
```
and try
```python
import dtapy
```
voila!
