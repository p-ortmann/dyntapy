#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from numba import config
config.DISABLE_JIT=1
#  without jit disabled numba cache needs to be cleared between runs since
# globals cannot be overwritten after compilation
from dyntapy.network_data import get_from_ox_and_save
from dyntapy.sta.assignment_methods import SUN
from dyntapy.demand import generate_random_bush
from dyntapy.visualization import show_network, show_demand
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.settings import static_parameters
(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_random_bush(g.number_of_nodes(), 3, seed=2)  # initiating bush with 3 random branches
obj = StaticAssignment(g, rand_od)
show_demand(obj)
SUN(g, od_matrix=rand_od)
val=static_parameters.assignment.logit_theta
show_network(g, title=f'GENT assignment, theta = {val}')
SUN(g, od_matrix=rand_od)
static_parameters.assignment.logit_theta =2
SUN(g, od_matrix=rand_od)
show_network(g, title=f'GENT assignment, theta = {2}')
static_parameters.assignment.logit_theta =1
SUN(g, od_matrix=rand_od)
show_network(g, title=f'GENT assignment, theta = {1}')
static_parameters.assignment.logit_theta =0.5
SUN(g, od_matrix=rand_od)
show_network(g, title=f'GENT assignment, theta = {0.5}')
static_parameters.assignment.logit_theta =0.1
SUN(g, od_matrix=rand_od)
show_network(g, title=f'GENT assignment, theta = {0.1}')
static_parameters.assignment.logit_theta =0.01
SUN(g, od_matrix=rand_od)
show_network(g, title=f'GENT assignment, theta = {0.01}')