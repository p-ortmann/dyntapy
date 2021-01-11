#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
from stapy.network_data import get_from_ox_and_save
from stapy.assignment_methods import DUE
from stapy.demand import generate_od_fixed
from visualization import plot_network, show_desire_lines
from stapy.assignment import  StaticAssignment
(g,deleted)= get_from_ox_and_save('Gent', reload=True)
#g = load_pickle('Gent')
print(f'number of nodes{g.tot_nodes()}')
# rand_od = generate_od(g.number_of_nodes(), origins_to_nodes_ratio=0.003)
rand_od = generate_od_fixed(g.tot_nodes(), 20)
#plot_network(deleted, mode='deleted elements')
obj=StaticAssignment(g,rand_od)
show_desire_lines(obj)
DUE(g, od_matrix=rand_od, method='dial_b') # methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
plot_network(g)
#show_convergence(g)
