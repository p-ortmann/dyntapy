#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
# this file tests warm-started dial-b
from stapy.algorithms.deterministic.dial_algorithm_B.bush_manager import dial_b
from stapy.assignment import StaticAssignment
from stapy.demand import generate_od_fixed
from stapy.network_data import get_from_ox_and_save
import numpy as np
(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_od_fixed(g.number_of_nodes(), 10)
obj = StaticAssignment(g, rand_od)  # initialization of the assignment object
print('static assignment object generated')
obj.link_travel_times, obj.link_flows, state = dial_b(obj)
obj.write_back()
obj.demand_dict[322][0] = np.append(obj.demand_dict[322][0], 6397)
obj.demand_dict[322][1] = np.append(obj.demand_dict[322][1], 500)
print('trying to update bush')
state.update_bushes(obj.demand_dict)
print('state updated successfully')
#assert obj.demand_dict != state.demand_dict
obj.link_travel_times, obj.link_flows, state = dial_b(obj, state)
print('hello')
