#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from network_data import create_cascetta_nw
from assignment_methods import DUE
from visualization import plot_network
from od_matrix_generator import generate_od_fixed
g= create_cascetta_nw()
rand_od=generate_od_fixed(g.number_of_nodes(), 5)
#DUE(g, od_matrix=rand_od, method='dial_b')
DUE(g, od_matrix=rand_od, method='dial_b')
plot_network(g)