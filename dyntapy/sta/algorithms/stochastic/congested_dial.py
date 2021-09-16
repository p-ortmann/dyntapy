#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dyntapy.sta.assignment import StaticAssignment
from numba import njit, jit
from numba.typed import Dict, List
from dyntapy.sta.algorithms.graph_utils import __shortest_path, __pred_to_epath2, make_out_links, make_in_links
from dyntapy.sta.algorithms.helper_funcs import __valid_edges, __topological_order, __bpr_cost
from dyntapy.settings import static_parameters
import numpy as np
from dyntapy.sta.algorithms.stochastic.uncongested_dial import load_all_bushes, generate_bushes

# as far as I recall this isn't working yet...
# missing a fix on the the topological order
# gotta add a routine for that, maybe a deterministic assignment ?
# TODO: testing and fixes ...
def congested_stochastic_assignment(obj: StaticAssignment):
    theta = static_parameters.assignment.logit_theta
    msa_max_iterations = static_parameters.assignment.msa_max_iterations
    msa_delta = static_parameters.assignment.msa_delta
    # print(f'my theta is {theta}')
    topological_orders, edges, L, largest_destination_labels = generate_bushes(obj.link_ff_times, obj.edge_map,
                                                                               obj.out_links,
                                                                               obj.demand, obj.tot_nodes)
    new_flows = load_all_bushes(obj, topological_orders, edges, L, largest_destination_labels, obj.link_ff_times, theta)
    gap = 1
    k = 1
    multiphase_counter = 0
    for bush in obj.demand.to_destinations.get_nnz_rows():
        largest_destination_labels[bush] = obj.tot_nodes
    while True:
        k = k + 1
        costs = __bpr_cost(new_flows, obj.link_capacities, obj.link_ff_times)
        new_edges = Dict()
        for bush in obj.demand.to_destinations.get_nnz_rows():
            L[bush], _ = __shortest_path(costs=costs,
                                         forward_star=make_out_links(edges[bush], len(topological_orders[bush])),
                                         edge_map=obj.edge_map, source=bush,
                                         targets=np.empty(0), node_order=obj.tot_nodes)
            topological_orders[bush] = __topological_order(L[bush])
            label = Dict()
            for j in topological_orders[bush]:
                label[topological_orders[bush][j]] = j
            bush_edges = List()
            for edge in edges[bush]:
                if label[edge[0]] < label[edge[1]]:
                    bush_edges.append(edge)
            new_edges[bush] = bush_edges
            assert bush in edges
            assert bush in topological_orders
        old_flows = new_flows.copy()
        new_flows = load_all_bushes(obj, topological_orders, new_edges, L, largest_destination_labels, costs, theta)
        nconverged = True
        for new_flow, old_flow in zip(new_flows, old_flows):
            if np.abs(new_flow - old_flow) / (
                    old_flow) < 0.001:
                continue
            else:
                print(np.abs(new_flow - old_flow) / (
                    old_flow))
                nconverged = False
        new_flows = 1 / k * new_flows + (k - 1) / k * old_flows
        converged = nconverged or k == msa_max_iterations
        multiphase_counter += 1
        if converged:
            break
        else:
            print(f'remaining gap is , multiphase stage: {multiphase_counter}')
        if multiphase_counter == 1000:
            k = 1
            multiphase_counter = 0

    return new_flows, __bpr_cost(new_flows, obj.link_capacities, obj.link_ff_times)
