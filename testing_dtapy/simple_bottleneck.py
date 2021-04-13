#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from tutorials.toy_networks import get_toy_network
from dtapy.demand import add_centroids_to_graph
import numpy as np
from dtapy.visualization import show_network, show_assignment
from dtapy.network_data import relabel_graph
from dtapy.core.time import SimulationTime

g = get_toy_network('simple_bottleneck')
centroid_x = np.array([1, 6])
centroid_y = np.array([1, 1])
g = add_centroids_to_graph(g, centroid_x, centroid_y)  # also adds connectors automatically
g = relabel_graph(g)  # adding link and node ids, connectors and centroids
# are the first elements
show_network(g, toy_network=True, title='Simple bottleneck with two centroids')
