#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from stapy.network_data import get_from_ox_and_save
from stapy.assignment_methods import SUN
from stapy.demand import generate_random_bush
from visualization import plot_network, show_desire_lines
from stapy.assignment import StaticAssignment
from stapy.settings import assignment_parameters
(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_random_bush(g.number_of_nodes(), 3, seed=2)  # initiating bush with 3 random branches
obj = StaticAssignment(g, rand_od)
show_desire_lines(obj)
SUN(g, od_matrix=rand_od)
val= assignment_parameters['logit_theta']
plot_network(g, title=f'GENT assignment, theta = {val}')
SUN(g, od_matrix=rand_od)
assignment_parameters['logit_theta']=2
SUN(g, od_matrix=rand_od)
plot_network(g, title=f'GENT assignment, theta = {2}')
assignment_parameters['logit_theta']=1
SUN(g, od_matrix=rand_od)
plot_network(g, title=f'GENT assignment, theta = {1}')
assignment_parameters['logit_theta']=0.5
SUN(g, od_matrix=rand_od)
plot_network(g, title=f'GENT assignment, theta = {0.5}')
assignment_parameters['logit_theta']=0.1
SUN(g, od_matrix=rand_od)
plot_network(g, title=f'GENT assignment, theta = {0.1}')
assignment_parameters['logit_theta']=0.01
SUN(g, od_matrix=rand_od)
plot_network(g, title=f'GENT assignment, theta = {0.01}')