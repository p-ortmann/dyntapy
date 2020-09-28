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
from numba import njit
def network_loading(obj:StaticAssignment):



    return link_travel_times, link_flows
def __initial_loading(edge_order, link_capacities, link_ff_times, edge_map, forward_star, demand_dict, node_order):
    flows = np.zeros(edge_order)
    bush_flows = Dict()
    topological_orders = Dict()
    edges = Dict()
    for bush in demand_dict:
        bush_flows[i] = np.zeros(edge_order)
        costs = __bpr_cost(capacities=link_capacities, ff_tts=link_ff_times, flows=flows)
        destinations = demand_dict[i][0]
        demands = demand_dict[i][1]
        distances, pred = __shortest_path(costs=costs, forward_star=forward_star, edge_map=edge_map, source=i,
                                          targets=np.empty(0), node_order=node_order)
        paths = __pred_to_epath2(pred, i, destinations, edge_map)
        topological_orders[i] = __topological_order(distances)
        label = Dict()
        for j in topological_orders[i]:
            label[topological_orders[i][j]] = j
        edges[i] = __valid_edges(edge_map, label)
        for path, path_flow in zip(paths, demands):
            for link_id in path:
                flows[link_id] += path_flow
                bush_flows[i][link_id] += path_flow

    return flows, bush_flows, topological_orders, edges
def set_labels():
    pass
def load_traffic(labels):
    pass