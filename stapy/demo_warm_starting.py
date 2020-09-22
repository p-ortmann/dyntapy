#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
# this file demonstrates how warm-started dial-b can be used.
from stapy.algorithms.dial_algorithm_B.bush_manager import dial_b
from stapy.assignment import StaticAssignment
from stapy.demand import generate_od_fixed
from stapy.network_data import get_from_ox_and_save
from stapy.visualization import plot_network, show_desire_lines

(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_od_fixed(g.number_of_nodes(), 10)
obj = StaticAssignment(g, rand_od)  # initialization of the assignment object
print('static assignment object generated')
show_desire_lines(obj)
obj.link_travel_times, obj.link_flows, state = dial_b(obj)
obj.write_back()
plot_network(obj.g, show_internal_ids=True)
print(obj.demand_dict)
