#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import os
os.environ['NUMBA_DISABLE_JIT']='1'
from dtapy.core.time import SimulationTime
g=SimulationTime(0,1,.025)
print(g.tot_time_steps)
