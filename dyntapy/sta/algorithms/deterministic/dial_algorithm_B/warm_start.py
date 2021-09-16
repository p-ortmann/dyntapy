#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from numba import njit
from numba.typed import List, Dict
from dyntapy.sta.algorithms.graph_utils import make_in_links
from dyntapy.sta.demand import Demand


# TODO: fix warm starting to work with the Demand class
class DialBResults:

    def __init__(self, demand_dict, flows, bush_flows, topological_orders, adjacency, edge_map):
        self.demand_dict = Dict()
        # there's no deep copy function in numba, so this it how it has to be ..
        for bush in demand_dict:
            my_list = List()
            my_list.append(demand_dict[bush][0])
            my_list.append(demand_dict[bush][1])
            self.demand_dict[bush] = my_list
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
                       self.edge_map, self.flows)


# @njit
def _update_bushes(new_demand_dict, old_demand_dict, adjacency, topological_orders, bush_flows, edge_map, flows):
    for bush in new_demand_dict:
        bush_backward_star = make_in_links(adjacency[bush],
                                           number_of_nodes=len(topological_orders[bush]))
        new_demands = new_demand_dict[bush][1]
        new_destinations = new_demand_dict[bush][0]
        old_demands = old_demand_dict[bush][1]
        old_destinations = old_demand_dict[bush][0]
        for new_id, new_destination in enumerate(new_destinations):
            old_id = -1
            previous_destination = False
            for id, old_destination in enumerate(old_destinations):
                if new_destination == old_destination:
                    previous_destination = True
                    old_id = id
                    break
            if previous_destination:
                delta_demand = new_demands[new_id] - old_demands[old_id]
            else:
                # previously no demand to this destination
                delta_demand = new_demands[new_id]
            while abs(delta_demand) > 0.00001:
                # print(f'{delta_demand=} ')
                edge_path, path_flow = get_max_flow_path(bush_backward_star, bush_flows[bush], edge_map,
                                                         new_destination)
                # print(f'{edge_path=} and {path_flow=}')
                delta_demand = update_path_flow(edge_path, delta_demand, path_flow, bush_flows[bush], flows)


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
        j = max_i
    return path, path_flow


# @njit
def update_path_flow(edge_path, delta_demand, path_flow, bush_flow, flows):
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
        flows[edge] += shift
    return delta_demand
