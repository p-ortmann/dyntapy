#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from network_data import get_from_ox_and_save
from od_matrix_generator import generate_od_fixed
from assignment_methods import DUE
from duet.io import nx_to_geojson
from visualization import plot_network

g, _ = get_from_ox_and_save('gent')
rand_od = generate_od_fixed(g.number_of_nodes(), 30)
# plot_network(deleted, mode='deleted elements')
# show_desire_lines(g, od_matrix=rand_od)
DUE(g, od_matrix=rand_od, method='dial_b')
plot_network(g)
nx_to_geojson(g, to_file=True)
