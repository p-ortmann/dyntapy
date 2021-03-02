#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from demand import parse_demand, generate_od_xy, DynamicDemand
from network_data import load_pickle
from assignment import Assignment
from settings import parameters
from core.assignment_cls import SimulationTime
import numpy as np

step_size = parameters.network_loading.step_size
# loading from data folder, assumes road_network_and_centroids was run previously
g = load_pickle('gent_grid_centroids')
geo_jsons = [generate_od_xy(20, 'Gent', seed=seed) for seed in [0, 1, 2]]
times = np.arange(2)
trip_graphs = {time: parse_demand(geo_json, g, time) for geo_json, time in zip(geo_jsons, times)}
# time unit is assumed to be hours, see parse demand
dynamic_demand = DynamicDemand(trip_graphs)
# convert everything to internal representations and parse
Assignment(g, dynamic_demand, SimulationTime(np.float32(0.0), np.float32(24.0), step_size=step_size))
# TODO: add tests for multi-edge parsing