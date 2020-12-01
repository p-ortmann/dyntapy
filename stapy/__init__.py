#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import os
from pathlib import Path
results_folder='results'
data_folder='data'
Path(os.getcwd()+"/"+data_folder).mkdir(parents=True, exist_ok=True)
Path(os.getcwd()+"/"+results_folder).mkdir(parents=True, exist_ok=True)
