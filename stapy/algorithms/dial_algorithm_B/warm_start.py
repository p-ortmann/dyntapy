#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import numpy as np
from numba import njit
from numba.typed import List
from stapy.algorithms.graph_utils import make_forward_stars, make_backward_stars


class DialBResults:
    def __init__(self, demand_dict, flows, bush_flows, topological_orders, adjacency, edge_map):
        self.demand_dict = demand_dict
        self.flows = flows
        self.bush_flows = bush_flows
        self.topological_orders = topological_orders
        self.adjacency = adjacency
        self.edge_map = edge_map

    def get_state(self):
        return self.flows, self.bush_flows, self.topological_orders, self.adjacency

    def update_bushes(self, new_demand_dict):
        """
        altered_demand_dict should only contain the bushes with changed demands, otherwise this could become costly ..
        """

        _update_bushes(new_demand_dict, self.demand_dict, self.adjacency, self.topological_orders, self.bush_flows,
                       self.edge_map)


@njit
def _update_bushes(new_demand_dict, old_demand_dict, adjacency, topological_orders, bush_flows, edge_map):
    for bush in new_demand_dict:
        bush_backward_star = make_backward_stars(adjacency[bush],
                                                 number_of_nodes=len(topological_orders[bush]))
        new_demands = new_demand_dict[bush][1]
        new_destinations = new_demand_dict[bush][0]
        old_demands = old_demand_dict[bush][1]
        old_destinations = old_demand_dict[bush][0]
        for id1, target in enumerate(new_destinations):
            if target in old_destinations:
                id2 = np.argwhere(target, new_destinations)[0][0]
                delta_demand = new_demands[id1] - old_demands[id2]
            else:
                # previously no demand to this destination
                delta_demand = new_demands[id1]
            while abs(delta_demand) > 0:
                edge_path, path_flow = get_max_flow_path(bush_backward_star, bush_flows[bush], edge_map,
                                                         delta_demand, target)
                delta_demand = update_path_flow(edge_path, delta_demand, path_flow, bush_flows[bush])


@njit
def get_max_flow_path(bush_backward_star, bush_flow, edge_map, target):
    j = target
    path_flow = 100000
    path = List()
    while len(bush_backward_star[j] > 0):
        max_flow = 0
        for i in bush_backward_star[j]:
            if bush_flow[edge_map[(i, j)]] >= max_flow:
                max_flow = bush_flow[edge_map[(i, j)]]
                max_i = i
        path.append(edge_map[(max_i, j)])
        path_flow = min(max_flow, path_flow)
    return path_flow, path


@njit
def update_path_flow(edge_path, delta_demand, path_flow, bush_flow):
    if delta_demand >= 0:
        shift = delta_demand
        delta_demand = 0
        # loading all traffic on max path
    else:
        shift = max(delta_demand, -path_flow)
        delta_demand = delta_demand - shift
        # either removes all traffic from max path or
        # removes delta_demand from max path if the flow on max path is larger than abs(delta_demand)
    for edge in edge_path:
        bush_flow[edge] += shift
    return delta_demand
