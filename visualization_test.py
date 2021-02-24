#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import networkx as nx
import numpy as np
from stapy.network_data import get_from_ox_and_save, load_pickle
from stapy.assignment_methods import DUE
from stapy.demand import generate_od_fixed
from visualization import show_assignment
from stapy.network_data import __filepath
from stapy.assignment import  StaticAssignment

def static_flow_to_dynamic_flow(val):
    middle = round(18 / 2)
    factors = np.arange(0.0, 1.0, 1.0 / middle)
    factors = np.concatenate((np.append(factors, 1), np.flip(factors)), axis=None)
    flows = [val * factor for factor in factors]
    return flows

def make_artificial_dynamic_graph(g: nx.DiGraph):
    for u, v, data in g.edges.data():
        data['flow'] = static_flow_to_dynamic_flow(data['flow'])
        data['costs'] = static_flow_to_dynamic_flow(data['costs'])

#(g,deleted)= get_from_ox_and_save('Gent')
g = load_pickle('Gent')
# rand_od = generate_od(g.number_of_nodes(), origins_to_nodes_ratio=0.003)
rand_od = generate_od_fixed(g.number_of_nodes(), 20)
#plot_network(deleted, mode='deleted elements')
#obj=StaticAssignment(g,rand_od)
DUE(g, od_matrix=rand_od, method='dial_b') # methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
make_artificial_dynamic_graph(g)
path = 'stapy/data/viz_test.pickle'
show_assignment(g)






