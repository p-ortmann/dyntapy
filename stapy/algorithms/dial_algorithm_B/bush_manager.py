#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
# There are no extensive comments in this module although they definitely would be helpful if looking at this for
# educational purposes, I'd recommend to consult Steven Boyles upcoming Book Transport Network Analysis and related
# course notes in github (https://sboyles.github.io/teaching/ce392c/10-bushbased.pdf)  for background and
# explanations and the more technical paper from Dial (Algorithm B: Accurate Traffic Equilibrium (and How to Bobtail
# Frank-Wolfe) as a reference and for understanding the more nitty-gritty details of implementation.
from stapy.algorithms.dial_algorithm_B.equilibrate_bush import __equilibrate_bush
from stapy.algorithms.dial_algorithm_B.utilities import __valid_edges, __topological_order
from stapy.assignment import StaticAssignment as StaticAssignment
import numpy as np
from stapy.algorithms.helper_funcs import __bpr_cost, __bpr_derivative
from stapy.algorithms.graph_utils import __shortest_path, __pred_to_epath2, make_backward_stars, make_forward_stars
from numba.typed import Dict
from numba import njit
from stapy.settings import assignment_parameters
from stapy.algorithms.dial_algorithm_B.warm_start import DialBResults

epsilon = assignment_parameters['dial_b_cost_differences']


def dial_b(obj: StaticAssignment, results: DialBResults = None):
    if results is None:
        flows, bush_flows, topological_orders, adjacency = __initial_loading(obj.edge_order,
                                                                         obj.link_capacities,
                                                                         obj.link_ff_times,
                                                                         obj.edge_map,
                                                                         obj.forward_star,
                                                                         obj.demand_dict,
                                                                         obj.node_order)
    else:
        flows, bush_flows, topological_orders, adjacency = results.get_state()
    costs = __bpr_cost(capacities=obj.link_capacities, ff_tts=obj.link_ff_times, flows=flows)
    derivatives = __bpr_derivative(flows=flows, capacities=obj.link_capacities, ff_tts=obj.link_ff_times)
    print('Dial equilibration started')
    while True:
        convergence_counter = 0
        for bush in obj.demand_dict:
            # creating a bush for each origin
            bush_forward_star = make_forward_stars(adjacency[bush], number_of_nodes=len(topological_orders[bush]))
            bush_backward_star = make_backward_stars(adjacency[bush], number_of_nodes=len(topological_orders[bush]))
            token = 0
            while True:
                # print(f'equilibrating bush {bush}')
                flows, bush_flows[bush], adjacency[bush], converged_without_shifts, L, U, bush_forward_star, \
                bush_backward_star = __equilibrate_bush(costs,
                                                        bush_flows=bush_flows[bush],
                                                        origin=bush, flows=flows,
                                                        edge_map=obj.edge_map,
                                                        topological_order=
                                                        topological_orders[bush],
                                                        derivatives=derivatives,
                                                        bush_adjacency_list=
                                                        adjacency[bush],
                                                        capacities=obj.link_capacities,
                                                        ff_tts=obj.link_ff_times,
                                                        bush_forward_star=bush_forward_star,
                                                        bush_backward_star=bush_backward_star,
                                                        epsilon=epsilon, global_forward_star=obj.forward_star)
                if converged_without_shifts:
                    if token >= 1:
                        if token == 1:
                            convergence_counter += 1
                        # print(f' no shifts after edges added for bush {bush}, moving on')
                        break

                for k in topological_orders[bush]:
                    assert k in L
                topological_orders[bush] = __topological_order(L)
                edges_added = __add_edges(L=L, bush_edges=adjacency[bush], costs=costs,
                                          edge_map=obj.edge_map, bush_forward_star=bush_forward_star,
                                          bush_backward_star=bush_backward_star,
                                          topological_order=topological_orders[bush])
                if not edges_added:
                    if converged_without_shifts and token == 0:
                        convergence_counter += 1
                    # print(f' bush {bush} is converged and no edges were added, moving on')
                    break
                else:
                    pass
                    # print('edges added')
                token += 1
        # print(f'number of converged bushes {convergence_counter} out of {len(obj.demand_dict)}')
        if convergence_counter == len(obj.demand_dict):
            break
    state = DialBResults(obj.demand_dict, flows, bush_flows, topological_orders, adjacency, obj.edge_map)
    return costs, flows, state




@njit
def __add_edges(L, bush_edges, costs, edge_map, bush_forward_star, bush_backward_star, topological_order):
    new_edges_added = False
    counter = 0
    new_edges = []
    for edge in edge_map:
        i = edge[0]
        j = edge[1]
        if L[i] + costs[edge_map[(i, j)]] < L[j] - epsilon:
            # pos2 = np.argwhere(topological_order == j)[0][0]
            # pos1 = np.argwhere(topological_order == i)[0][0]
            # assert pos2>pos1
            # if pos1 > pos2:
            #    print('something wrong with labels or shifting')
            bush_edges.append((i, j))
            bush_backward_star[j] = np.append(bush_backward_star[j], i)
            bush_forward_star[i] = np.append(bush_forward_star[i], j)
            counter += 1
            new_edges_added = True
            new_edges.append((i, j))

    return new_edges_added


@njit
def __initial_loading(edge_order, link_capacities, link_ff_times, edge_map, forward_star, demand_dict, node_order):
    flows = np.zeros(edge_order)
    bush_flows = Dict()
    topological_orders = Dict()
    edges = Dict()
    for i in demand_dict:
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
