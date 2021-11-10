#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

# from numba import config
# config.DISABLE_JIT = 1
from dyntapy.demand import parse_demand, generate_od_xy, DynamicDemand
from dyntapy.network_data import load_pickle
from dyntapy.dta.assignment import Assignment
from dyntapy.dta.core.time import SimulationTime
import numpy as np
from dyntapy.settings import default_dynamic_city as city
from dyntapy.visualization import show_demand, show_dynamic_network
from testing.road_network_and_centroids import get_graph

def init_assignment():
    # loading from data folder, assumes road_network_and_centroids was run previously
    try:
        g = load_pickle(city + '_grid_centroids')
    except NameError:
        g = get_graph(city)
    geo_jsons = [generate_od_xy(7, city, seed=seed, max_flow=200) for seed in [1, 2]]
    times = np.arange(2)
    trip_graphs = [parse_demand(geo_json, g, time) for geo_json, time in zip(geo_jsons, times)]
    for trip_graph, time in zip(trip_graphs, times):
        pass
        # show_demand(trip_graph, title=f'demand at {time}')
    # time unit is assumed to be hours, see parse demand
    dynamic_demand = DynamicDemand(trip_graphs, times)
    # convert everything to internal representations and parse
    assignment = Assignment(g, dynamic_demand, SimulationTime(np.float32(0.0), np.float32(2.0),np.float32(0.25)))
    # TODO: add tests for multi-edge parsing
    flows, costs = assignment.run(method='incremental_assignment'
                                         )
    show_dynamic_network(g, SimulationTime(np.float32(0.0), np.float32(2.0), np.float32(0.25)),
                         link_kwargs={'flows': flows, 'costs': costs}, show_nodes=False)

if __name__ == '__main__':
    init_assignment()
