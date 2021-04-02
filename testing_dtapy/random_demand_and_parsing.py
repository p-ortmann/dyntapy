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
from dtapy.visualization import show_demand

step_size = parameters.network_loading.step_size
# loading from data folder, assumes road_network_and_centroids was run previously
g = load_pickle(city + '_grid_centroids')
geo_jsons = [generate_od_xy(4, city, seed=seed) for seed in [0, 1, 2]]
times = np.arange(2)
trip_graphs = {time: parse_demand(geo_json, g, time) for geo_json, time in zip(geo_jsons, times)}
# time unit is assumed to be hours, see parse demand
dynamic_demand = DynamicDemand(trip_graphs)
# convert everything to internal representations and parse
assignment = Assignment(g, dynamic_demand, SimulationTime(np.float32(0.0), np.float32(2.0), step_size=step_size))
# TODO: add tests for multi-edge parsing
methods = assignment.get_methods()
assignment.run(methods.i_ltm_aon)
# show_demand(trip_graphs[0])
# show_demand([trip_graphs[1]])
# show_demand(trip_graphs[2])
print('ran successfully')
