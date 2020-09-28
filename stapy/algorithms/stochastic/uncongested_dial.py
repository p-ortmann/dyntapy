#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# the
from stapy.assignment import StaticAssignment
from stapy.settings import assignment_parameters
from numba import njit
from numba.typed import Dict, List
from stapy.algorithms.graph_utils import __shortest_path, __pred_to_epath2
from stapy.algorithms.helper_funcs import __valid_edges, __topological_order
import numpy as np

theta = assignment_parameters['logit_theta']


def uncongested_stochastic_assignment(obj: StaticAssignment):
    flows=np.zeros(obj.edge_order)
    topological_orders, edges, L = generate_bushes(obj.link_ff_times, obj.edge_map, obj.forward_star,
                                                   obj.demand_dict, obj.node_order)
    for bush in obj.demand_dict:
        load_bush(obj.edge_map, obj.demand_dict, topological_orders[bush], edges[bush], L[bush], flows)


def generate_bushes(link_ff_times, edge_map, forward_star, demand_dict, node_order):
    topological_orders = Dict()
    edges = Dict()
    assert len(demand_dict) > 0  # demand dict is not empty..
    for bush in demand_dict:
        destinations = demand_dict[bush][0]
        L, pred = __shortest_path(costs=link_ff_times, forward_star=forward_star, edge_map=edge_map, source=i,
                                  targets=np.empty(0), node_order=node_order)
        paths = __pred_to_epath2(pred, bush, destinations, edge_map)
        topological_orders[bush] = __topological_order(L)
        label = Dict()
        for j in topological_orders[bush]:
            label[topological_orders[bush][j]] = j
        edges[bush] = __valid_edges(edge_map, label)

    return topological_orders, edges, L


def load_bush(edge_map, costs, demand_dict, topological_order, bush_edges, L, flows):
    return flows
