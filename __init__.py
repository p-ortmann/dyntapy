import os
import sys
from pathlib import Path
sys.path.append(os.getcwd()+os.path.sep)

results_folder='results'
data_folder='data'
Path(os.getcwd()+"/"+data_folder).mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder).mkdir(parents=True, exist_ok=True)