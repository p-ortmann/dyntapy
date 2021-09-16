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
from dyntapy.network_data import get_from_ox_and_save
from dyntapy.sta.assignment_methods import DUE
from dyntapy.demand import generate_od_fixed
from dyntapy.visualization import show_network, show_demand
from dyntapy.sta.assignment import  StaticAssignment
(g,deleted)= get_from_ox_and_save('Gent', reload=True)
#g = load_pickle('Gent')
print(f'number of nodes{g.number_of_nodes()}')
# rand_od = generate_od(g.number_of_nodes(), origins_to_nodes_ratio=0.003)
rand_od = generate_od_fixed(g.number_of_nodes(), 20)
#show_network(deleted, mode='deleted elements')
obj=StaticAssignment(g,rand_od)
show_demand(obj)
DUE(g, od_matrix=rand_od, method='dial_b') # methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
show_network(g)
#show_convergence(g)
