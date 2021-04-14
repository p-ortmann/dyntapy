import os
import sys
from pathlib import Path
from dtapy.settings import numba_config

sys.path.append(os.getcwd() + os.path.sep)

results_folder = 'results'
data_folder = 'data'

Path(os.getcwd() + os.path.sep + data_folder).mkdir(parents=True, exist_ok=True)
Path(os.getcwd() + os.path.sep + results_folder).mkdir(parents=True, exist_ok=True)

for key, value in numba_config.items():
    os.environ[key] = value


current_network = None
