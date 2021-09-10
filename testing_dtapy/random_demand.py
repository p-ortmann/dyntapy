#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from numba import config
config.DISABLE_JIT=1
from dtapy.demand import parse_demand, generate_od_xy, DynamicDemand
from dtapy.network_data import load_pickle
from dtapy.assignment import Assignment
from dtapy.settings import parameters
from dtapy.core.time import SimulationTime
import numpy as np
from dtapy.settings import default_city as city
from dtapy.visualization import show_demand, show_dynamic_network


def init_assignment():
    step_size = parameters.network_loading.step_size
    # loading from data folder, assumes road_network_and_centroids was run previously
    g = load_pickle(city + '_grid_centroids')
    geo_jsons = [generate_od_xy(1, city, seed=seed, max_flow=500) for seed in [1, 2]]
    times = np.arange(2)
    trip_graphs = [parse_demand(geo_json, g, time) for geo_json, time in zip(geo_jsons, times)]
    for trip_graph, time in zip(trip_graphs, times):
        pass
        #show_demand(trip_graph, title=f'demand at {time}')
    # time unit is assumed to be hours, see parse demand
    dynamic_demand = DynamicDemand(trip_graphs, times)
    # convert everything to internal representations and parse
    assignment = Assignment(g, dynamic_demand, SimulationTime(np.float32(0.0), np.float32(2.0), step_size=step_size))
    # TODO: add tests for multi-edge parsing
    flows, costs = assignment.run(method='i_ltm_aon')
    show_dynamic_network(g, SimulationTime(np.float32(0.0), np.float32(2.0), step_size=step_size),
                         link_kwargs={'flows': flows, 'costs': costs}, show_nodes=False)


if __name__=='__main__':
    init_assignment()

