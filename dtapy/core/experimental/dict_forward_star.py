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
import numba as nb
import numpy as np
from collections import OrderedDict


@nb.njit
def make_forward_stars(edge_array, number_of_nodes):
    forward_star = nb.typed.Dict()
    star_sizes = np.zeros(number_of_nodes, dtype=nb.int64)
    nodes = np.arange(number_of_nodes, dtype=nb.int64)
    for i in nodes:
        forward_star[i] = np.empty(10,
                                   dtype=nb.int64)  # In traffic networks nodes will never have more than 10 outgoing edges..
    for row in edge_array:
        i, j = row[0], row[1]
        forward_star[i][star_sizes[i]] = j
        star_sizes[i] += 1
    for i in nodes:
        forward_star[i] = forward_star[i][:star_sizes[i]]
    return forward_star


