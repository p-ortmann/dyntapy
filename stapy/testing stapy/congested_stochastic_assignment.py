#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from stapy.network_data import get_from_ox_and_save
from stapy.assignment_methods import SUE
from stapy.demand import generate_od_fixed
from stapy.visualization import plot_network, show_desire_lines
from stapy.assignment import StaticAssignment
from stapy.settings import assignment_parameters
(g, deleted) = get_from_ox_and_save('Gent')
rand_od = generate_od_fixed(g.number_of_nodes(),1)# initiating bush with 3 random branches
obj = StaticAssignment(g, rand_od)
show_desire_lines(obj)
SUE(g, od_matrix=rand_od)
val= assignment_parameters['logit_theta']
plot_network(g, title=f'GENT assignment, theta = {val}')

