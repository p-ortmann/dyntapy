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
from tutorials.toy_networks import get_toy_network
from dtapy.demand import add_centroids_to_graph
import numpy as np
from dtapy.visualization import show_network, show_assignment, show_demand
from dtapy.network_data import relabel_graph
from dtapy.demand import od_graph_from_matrix, DynamicDemand
from dtapy.core.time import SimulationTime
from dtapy.assignment import Assignment

g = get_toy_network('simple_bottleneck')
centroid_x = np.array([0, 7])
centroid_y = np.array([0.9, 0.9])
g = add_centroids_to_graph(g, centroid_x, centroid_y, toy_network=True)  # also adds connectors automatically
g = relabel_graph(g)  # adding link and node ids, connectors and centroids
# are the first elements
show_network(g, toy_network=True, title='Simple bottleneck with two centroids')
od_matrix = np.zeros(4).reshape((2, 2))
od_matrix[0, 1] = 400
od_graph = od_graph_from_matrix(od_matrix, centroid_x, centroid_y)
show_demand(od_graph, toy_network=True)
dynamic_demand = DynamicDemand([od_graph], insertion_times=[0])
# convert everything to internal representations and parse
simulation_time=SimulationTime(np.float32(0.0), np.float32(2.0), step_size=0.25)
assignment = Assignment(g, dynamic_demand, simulation_time)
methods = assignment.get_methods()
flows, costs = assignment.run(methods.i_ltm_aon)
show_assignment(g,simulation_time,toy_network=True,link_kwargs={'flows':flows, 'costs':costs} )
