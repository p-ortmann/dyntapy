#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from dyntapy.network_data import get_from_ox_and_save
from dyntapy.sta.assignment_methods import SUE
from dyntapy.demand import generate_od_fixed
from dyntapy.visualization import show_network, show_demand
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.settings import static_parameters
(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_od_fixed(g.number_of_nodes(),1)# initiating bush with 3 random branches
obj = StaticAssignment(g, rand_od)
show_demand(obj)
SUE(g, od_matrix=rand_od)
val= static_parameters.assignment.logit_theta
show_network(g, title=f'GENT assignment, theta = {val}')

