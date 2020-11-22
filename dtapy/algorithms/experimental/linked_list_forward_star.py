#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
import numba as nb

def relabel_nodes(edge_array):
    # for all edges it should hold that
    pass
def make_link_matrix(edge_array, number_of_nodes):
    edges = np.empty(shape=(len(edge_array), 5))
    nodes = np.empty(number_of_nodes)
    capacity=np.arange(len(edge_array)).transpose()
    edges[:,4]=capacity
    return nodes,edges
@nb.njit
def test_access_time(node):
    pass



