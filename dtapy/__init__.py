#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

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
