#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from numba import config
config.DISABLE_JIT = 1
from dyntapy.networks.get_networks import get_toy_network
from dyntapy.demand import add_centroids_to_graph
import numpy as np
from dyntapy.visualization import show_network, show_dynamic_network, show_demand
from dyntapy.network_data import relabel_graph
from dyntapy.demand import od_graph_from_matrix, DynamicDemand
from dyntapy.dta.core.time import SimulationTime
from dyntapy.dta.assignment import Assignment
from bokeh.plotting import figure, output_file, show


toy_network = 'simple_bottleneck'
g = get_toy_network(toy_network)
centroid_x = np.array([0, 7])
centroid_y = np.array([0.9, 0.9])
g = add_centroids_to_graph(g, centroid_x, centroid_y, euclidean=True)  # also adds connectors automatically
g = relabel_graph(g)  # adding link and node ids, connectors and centroids
# are the first elements
show_network(g, toy_network=True, title=toy_network)
od_matrix = np.zeros(4).reshape((2, 2))
demand_values = list(range(125, 145))
costs_on_link_two = []
time_interval = 0
for val in demand_values:
    od_matrix[0, 1] = val
    od_graph = od_graph_from_matrix(od_matrix, centroid_x, centroid_y)
    dynamic_demand = DynamicDemand([od_graph], insertion_times=[0])
    simulation_time=SimulationTime(np.float32(0.0), np.float32(2.0), step_size=0.25)
    assignment = Assignment(g, dynamic_demand, simulation_time)
    flows, costs = assignment.run()
    costs_on_link_two.append(costs[time_interval,2])
show_dynamic_network(g, simulation_time,flows = flows, toy_network=True)
output_file("cost_function_sensitivity.html")
p = figure(title = f'Cost sensitivity of link 2 to increasing demand at in time interval {time_interval}',
           width=400, height=400)
# add a line renderer
p.line(demand_values, costs_on_link_two, line_width=2)
p.xaxis.axis_label = 'Demand Left to Right'
p.yaxis.axis_label = 'Costs on Link 2'
show(p)