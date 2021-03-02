#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from demand import generate_od_xy, parse_demand, add_centroids_to_graph
from network_data import get_from_ox_and_save
from assignment import Assignment

(g, deleted) = get_from_ox_and_save('Gent', reload=True)
print(f'number of nodes{g.number_of_nodes()}')
add_centroids_to_graph('Gent', g)
print(f'number of nodes{g.number_of_nodes()} incl connectors')


assignment = Assignment(g)

print('init passed successfully')
print('hi')