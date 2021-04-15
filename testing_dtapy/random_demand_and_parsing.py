#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.demand import parse_demand, generate_od_xy, DynamicDemand
from dtapy.network_data import load_pickle
from dtapy.assignment import Assignment
from dtapy.settings import parameters
from dtapy.core.time import SimulationTime
import numpy as np
from dtapy.settings import default_city as city
from dtapy.visualization import show_demand, show_assignment
step_size = parameters.network_loading.step_size
# loading from data folder, assumes road_network_and_centroids was run previously
g = load_pickle(city + '_grid_centroids')
geo_jsons = [generate_od_xy(4, city, seed=seed, max_flow=500) for seed in [0, 1]]
times = np.arange(2)
trip_graphs = [parse_demand(geo_json, g, time) for geo_json, time in zip(geo_jsons, times)]
for trip_graph, time in zip(trip_graphs, times):
    show_demand(trip_graph, title=f'demand at {time}')
# time unit is assumed to be hours, see parse demand
dynamic_demand = DynamicDemand(trip_graphs, times)
# convert everything to internal representations and parse
assignment = Assignment(g, dynamic_demand, SimulationTime(np.float32(0.0), np.float32(2.0), step_size=step_size))
# TODO: add tests for multi-edge parsing
methods = assignment.get_methods()
flows, costs = assignment.run(methods.i_ltm_aon)
show_assignment(g, SimulationTime(np.float32(0.0),np.float32(2.0), step_size=step_size),
                link_kwargs={'flows': flows}, show_nodes=False)
print('ran successfully')
