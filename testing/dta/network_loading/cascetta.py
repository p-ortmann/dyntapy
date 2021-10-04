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
# from numba import config
# config.DISABLE_JIT=1
from dyntapy.networks.get_networks import get_toy_network
from dyntapy.demand import add_centroids_to_graph
import numpy as np
from dyntapy.visualization import show_network, show_dynamic_network, show_demand
from dyntapy.network_data import relabel_graph
from dyntapy.demand import od_graph_from_matrix, DynamicDemand
from dyntapy.dta.core.time import SimulationTime
from dyntapy.dta.assignment import Assignment

toy_network = 'cascetta'
g = get_toy_network('cascetta')
centroid_x = np.array([1, 7, 4])
centroid_y = np.array([1, 1, 3.5])
g = add_centroids_to_graph(g, centroid_x, centroid_y, euclidean=True)  # also adds connectors automatically
g = relabel_graph(g)  # adding link and node ids, connectors and centroids
# are the first elements
show_network(g, toy_network=True, title=toy_network)
od_matrix = np.zeros(9).reshape((3, 3))
od_matrix[0, 1] = 500
od_matrix[2, 1] = 500
od_graph = od_graph_from_matrix(od_matrix, centroid_x, centroid_y)
show_demand(od_graph, toy_network=True)
dynamic_demand = DynamicDemand([od_graph], insertion_times=[0])
# convert everything to internal representations and parse
simulation_time = SimulationTime(np.float32(0.0), np.float32(2.0), step_size=0.25)
assignment = Assignment(g, dynamic_demand, simulation_time)
flows, costs = assignment.run()
show_dynamic_network(g, simulation_time, toy_network=True, link_kwargs={'flows': flows, 'costs': costs},
                     title='assignment on ' + toy_network)
