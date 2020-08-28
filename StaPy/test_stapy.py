#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
from network_data import get_from_ox_and_save, load_pickle
from assignment_methods import DUE
from networkx import DiGraph
from od_matrix_generator import generate_od, generate_od_fixed
from visualization import plot_network, show_desire_lines, show_convergence
import osmnx as ox
#(g,deleted)= get_from_ox_and_save('Gent', reload=False)
g = load_pickle('Gent')
print(f'number of nodes{g.number_of_nodes()}')
# rand_od = generate_od(g.number_of_nodes(), origins_to_nodes_ratio=0.003)
rand_od = generate_od_fixed(g.number_of_nodes(), 35)
#plot_network(deleted, mode='deleted elements')
#show_desire_lines(g, od_matrix=rand_od)
DUE(g, od_matrix=rand_od, method='dial_b') # methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
plot_network(g)
#show_convergence(g)
