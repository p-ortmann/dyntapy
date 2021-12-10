#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from numba import config
# config.FULL_TRACEBACKS = 1
config.DISABLE_JIT = 1
from dyntapy.demand import parse_demand, generate_od_xy
from dyntapy.network_data import load_pickle
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.settings import default_dynamic_city as city
from dyntapy.visualization import show_demand, show_network
from dyntapy.sta.assignment_methods import DUN, DUE, SUN, SUE
from testing.road_network_and_centroids import get_graph
import numpy as np

try:
    g = load_pickle(city + '_grid_centroids')
except NameError:  # no data file
    g = get_graph(city)
seed = 0
json_demand = generate_od_xy(5, city, seed=seed, max_flow=600)
od_graph = parse_demand(json_demand, g, 0)
obj = StaticAssignment(g, od_graph)

method = 'dial_b'
flows, costs = DUE(obj, method)
show_network(g, flows)
print(f'{method=} ran successfully')

