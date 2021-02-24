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
from network_data import get_from_ox_and_save
from assignment import Assignment

(g,deleted)= get_from_ox_and_save('Gent')
print(f'number of nodes{g.number_of_nodes()}')
#plot_network(deleted, mode='deleted elements')
obj=Assignment(g, od_matrix=None)
