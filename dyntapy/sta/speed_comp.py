#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from dyntapy.network_data import load_pickle
import networkx as nx
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.demand import generate_od_fixed
from dyntapy.sta.algorithms.graph_utils import __shortest_path
import numpy as np
import igraph as ig
import timeit

g : nx.DiGraph = load_pickle('antwerp')
od_matrix=generate_od_fixed(g.number_of_nodes(), 0)
obj = StaticAssignment(g, od_matrix)
random_og= np.random.choice(np.arange(g.number_of_nodes()))
random_dest= np.random.choice(np.arange(g.number_of_nodes()))
og_nx, dest_nx=obj.node_map_to_nx[random_og], obj.node_map_to_nx[random_dest]
targetts=np.empty(0)
__shortest_path(obj.link_travel_times, obj.out_links, obj.edge_map, random_og, targetts, obj.tot_nodes)
my_igraph=ig.Graph(obj.edge_map.keys(), directed=True)
my_igraph.shortest_paths(source=random_og,weights=obj.link_ff_times, mode='OUT')
nx.shortest_path(g, source= og_nx, weight='weight')
my_weight_str=str('weight')
print(timeit.timeit(stmt='__shortest_path(obj.link_travel_times, obj.forward_star, obj.edge_map,random_og, targetts, '
                         'obj.node_order)', setup='from __main__ import __shortest_path, obj, targetts, random_og',number=100))
print(timeit.timeit(stmt='nx.shortest_path(g, source= og_nx, weight=my_weight_str)',setup='from __main__ import g, og_nx, my_weight_str,nx', number=100))
print(timeit.timeit(stmt='my_igraph.shortest_paths(source=random_og,weights=obj.link_ff_times)',setup='from __main__ '
                                                                                                      'import '
                                                                                                      'my_igraph, '
                                                                                                      'random_og, '
                                                                                                      'obj', number=100))
