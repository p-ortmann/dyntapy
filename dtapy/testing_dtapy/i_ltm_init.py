#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.demand import generate_od_xy, parse_demand, add_centroids_from_grid
from dtapy.network_data import get_from_ox_and_save
import numpy as np
from dtapy.assignment import Assignment
from dtapy.core.network_loading.link_models.i_ltm import i_ltm

(g, deleted) = get_from_ox_and_save('Gent', reload=True)
print(f'number of nodes{g.number_of_nodes()}')
add_centroids_from_grid('Gent', g)
print(f'number of nodes{g.number_of_nodes()} incl connectors')
gjsons = [generate_od_xy(20, 'Gent', seed=seed) for seed in np.arange(3) * 3]
time = 0
for gjson in gjsons:
    parse_demand(gjson, g, time)
    time = time + 1
assignment = Assignment(g)
i_ltm(assignment.network, assignment.dynamic_demand, assignment.results, assignment.time)

print('init passed successfully')
print('hi')
