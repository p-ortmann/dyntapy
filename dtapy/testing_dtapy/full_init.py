#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# testing initialization of all information needed for LTM
from dtapy.core.jitclasses import SimulationTime
from dtapy.demand import _build_demand, generate_od_fixed
from dtapy.network_data import get_from_ox_and_save
import numpy as np
from dtapy.assignment import Assignment

(g, deleted) = get_from_ox_and_save('Gent')
print(f'number of nodes{g.number_of_nodes()}')
start_time = 6  # time of day in hrs
end_time = 12
demands = [generate_od_fixed(g.number_of_nodes(), 20), generate_od_fixed(g.number_of_nodes(), 20)]
insertion_times = np.array([6, 7])
ltm_dt = 0.25  # ltm timestep in hrs
simulation_time = SimulationTime(start_time, end_time, ltm_dt)
demand_simulation = _build_demand(demands, insertion_times, simulation_time, g.number_of_nodes())
Assignment(g, demand_simulation, simulation_time)
print('init passed successfully')
