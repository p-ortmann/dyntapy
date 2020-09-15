#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import numpy as np
from numba import njit
from stapy.algorithms.graph_utils import make_forward_stars, make_backward_stars


class DialBResults:
    def __init__(self, demand_dict, flows, bush_flows, topological_orders, adjacency, edge_map):
        self.demand_dict=demand_dict
        self.flows = flows
        self.bush_flows = bush_flows
        self.topological_orders = topological_orders
        self.adjacency = adjacency
        self.edge_map=edge_map


    def get_state(self):
        return self.flows, self.bush_flows, self.topological_orders, self.adjacency

    def update_bushes(self, new_demand_dict):
        """
        altered_demand_dict should only contain the bushes with changed demands, otherwise this could become costly ..
        """

        for bush in new_demand_dict:
            bush_backward_star = make_backward_stars(self.adjacency[bush], number_of_nodes=len(self.topological_orders[bush]))
            new_demands = new_demand_dict[bush][1]
            new_destinations = new_demand_dict[bush][0]
            old_demands = self.demand_dict[bush][1]
            old_destinations = self.demand_dict[bush][0]
            for id1, target in enumerate(new_destinations):
                if target in old_destinations:
                    id2 = np.argwhere(target, new_destinations)[0][0]
                    delta_demand = new_demands[id1]-old_demands[id2]
                else:
                    # previously no demand to this destination
                    delta_demand = new_demands[id1]
                if delta_demand<0:
                    get_max_flow_path(bush_backward_star, bush_flows=)





@njit
def __update_bushes(new_demand_dict, old_demand_dict, adjacency, topological_orders, bush_flows, edge_map):
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
            if delta_demand < 0:
                while delta_demand<0:
                    get_max_flow_path(bush_backward_star, bush_flows[bush], edge_map, delta_demand, target)
                    update_path_flow
@njit
def update_path_flow():
    pass
@njit
def get_max_flow_path(bush_backward_star, bush_flow, edge_map):
    pass
    # for i in self.demand_dict:
    #     costs = __bpr_cost(capacities=link_capacities, ff_tts=link_ff_times, flows=flows)
    #     destinations = demand_dict[i][0]
    #     demands = demand_dict[i][1]
    #     self.
