#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from numba import njit
from numba.typed import List
from stapy.setup import int_dtype


@njit
def __valid_edges(edge_map, label):
    '''

    Parameters
    ----------
    edge_map : dictionary keyed by (i,j) with edge index as value
    distances : dictionary of distances keyed by node ids

    Returns
    -------

    '''
    edges = List()
    for (i, j) in edge_map:
        if label[j] > label[i]:
            edges.append((i, j))
    return edges


@njit
def __topological_order(distances):
    '''

    Parameters
    ----------
    distances : dictionary of distances keyed by node ids

    Returns
    -------

    '''
    topological_order = np.empty(len(distances), dtype=int_dtype)
    for counter, (node, dist) in enumerate(sorted(distances.items(), key=lambda x: x[1])):
        topological_order[counter] = node
    return topological_order