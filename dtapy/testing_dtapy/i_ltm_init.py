#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import SimulationTime
from dtapy.demand import _build_demand, generate_od_xy, parse_demand
from dtapy.network_data import get_from_ox_and_save
import numpy as np
from dtapy.assignment import Assignment
from dtapy.core.network_loading.i_ltm_setup import i_ltm_setup

(g, deleted) = get_from_ox_and_save('Gent')
print(f'number of nodes{g.number_of_nodes()}')
gjsons = [generate_od_xy(20, 'Gent',seed=seed) for seed in np.arange(3)*3]
for gjson in gjsons:
    parse_demand(gjson, g, matching_dist=3000)
print(f'number of nodes{g.number_of_nodes()} incl connectors')
assignment = Assignment(g)
print('init passed successfully')
print('hi')
