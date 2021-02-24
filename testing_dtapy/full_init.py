#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# testing initialization of all information needed for LTM
from core import SimulationTime
from demand import _build_internal_dynamic_demand, generate_od_xy, add_centroids_from_grid, parse_demand
from network_data import get_from_ox_and_save
import numpy as np
from assignment import Assignment
from settings import parameters
from visualization import show_demand
ltm_dt = parameters.network_loading.time_step

(g, deleted) = get_from_ox_and_save('Gent')
print(f'number of nodes{g.number_of_nodes()}')
start_time = 6  # time of day in hrs
end_time = 12
insertion_times = np.array([6, 7])
add_centroids_from_grid('Gent', g)
demands = [generate_od_xy(20,'Gent'), generate_od_xy(20,'Gent', seed=1)]
[parse_demand(demand,g ,t) for demand,t in zip(demands, insertion_times) ]
# demand is now stored under g.graph['od_graphs'], it's a dict of nx.DiGraphs with time as the key
# can visualized with
show_demand(g.graph['od_graphs'][6])
simulation_time = SimulationTime(start_time, end_time, ltm_dt)
demand_simulation = _build_internal_dynamic_demand(demands, insertion_times, simulation_time, g.number_of_nodes())
Assignment(g, simulation_time)
print('init passed successfully')
