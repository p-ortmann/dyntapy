#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#

from dyntapy.results import StaticResult
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.results import get_od_flows, get_selected_link_analysis
from dyntapy import show_demand, show_network, show_dynamic_network, \
    show_link_od_flows
from dyntapy import StaticAssignment
from dyntapy.demand_data import generate_od_xy, add_centroids, \
    auto_configured_centroids, parse_demand
from dyntapy.demand import SimulationTime
from dyntapy import get_shortest_paths, get_all_shortest_paths, get_toy_network
from dyntapy.dta.orca_nodel_model import orca_node_model
from dyntapy.assignments import DynamicAssignment
from dyntapy.demand_data import od_graph_from_matrix
from dyntapy.demand import DynamicDemand
import os
import pathlib
import sys
from pickle import dump, load

import numpy as np

one_up = pathlib.Path(__file__).parents[1]
sys.path.append(one_up.as_posix())

#  Part One Adding Centroids
g = get_toy_network('siouxfalls')
centroid_x = np.array([-96.8, -96.7])
centroid_y = np.array([43.6, 43.6])
g = add_centroids(g, centroid_x, centroid_y)

g = relabel_graph(g)  # adding link and node ids, connectors and centroids are the first elements
show_network(g, euclidean=True)

#  Part Two Defining Od matrix and dynamic demand
od_matrix = np.zeros(4).reshape((2, 2))
od_matrix[0, 1] = 2499
od_graph = od_graph_from_matrix(od_matrix, centroid_x, centroid_y)

show_demand(od_graph)
dynamic_demand = DynamicDemand([od_graph,od_graph,od_graph,od_graph], insertion_times=[0,0.25,0.5,1])

#  Part Three Defining simulation time
simulation_time = SimulationTime(
    np.float32(0.0), np.float32(2.0), step_size=0.25)

assignment = DynamicAssignment(g, dynamic_demand, simulation_time)

methods = ['incremental_assignment', 'i_ltm_aon']
for method in methods:
    result = assignment.run(method=method)
    print(result)
    show_dynamic_network(g, simulation_time,
                         flows=result.flows, euclidean=True)
