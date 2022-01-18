#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#

import numpy as np
from numba import njit
from numba.typed import Dict

from dyntapy.settings import parameters
from dyntapy.sta.utilities import __bpr_cost_single, __bpr_derivative_single
from dyntapy.graph_utils import get_link_id

epsilon = parameters.static_assignment.dial_b_cost_differences


# it should be possible to make this quite a bit faster still by
# removing all the dictionaries used for the labels


@njit
def __equilibrate_bush(
    costs,
    bush_flows,
    origin,
    flows,
    topological_order,
    derivatives,
    links_in_bush,
    capacities,
    ff_tts,
    bust_out_links,
    bush_in_links,
    epsilon,
    global_out_links,
    tot_links,
    to_node,
    from_node,
):
    # we equilibrate each bush to convergence before moving on to the next ..
    # once shifting in the bush has reached equilibrium
    # we try to add shortcut links from the graph
    # and start shifting again, until this doesn't yield improvements anymore.
    converged_without_shifts = False
    max_path_predecessors = Dict()
    min_path_predecessors = Dict()
    max_path_predecessors[origin] = -1
    min_path_predecessors[origin] = -1
    L, U, label = Dict(), Dict(), Dict()
    U[origin] = 0.0
    L[origin] = 0.0
    for index, node in enumerate(topological_order):
        label[node] = index
    max_delta_path_cost, L, U = __update_trees(
        1,
        len(topological_order),
        L,
        U,
        min_path_predecessors,
        max_path_predecessors,
        topological_order,
        costs,
        bush_flows,
        bush_in_links,
    )
    # print(f'________the remaining cost differences in this
    # bush for origin {origin} are {max_delta_path_cost}______')
    for i in topological_order:
        assert i in L

    if epsilon > max_delta_path_cost:
        converged_without_shifts = True
        # print(f'no shifts were ever necessary,
        # delta: {max_delta_path_cost} smaller than epsilon {epsilon}')
    while epsilon < max_delta_path_cost:
        # print(f'calling shift flow, cost differences are:
        # {max_delta_path_cost} larger than {epsilon} ')
        lowest_order_node = __shift_flow(
            topological_order,
            L,
            U,
            min_path_predecessors,
            max_path_predecessors,
            derivatives,
            costs,
            global_out_links,
            label,
            bush_flows,
            capacities,
            flows,
            ff_tts,
            bush_in_links,
        )
        # print(f'updating trees, branch node is: {lowest_order_node}')
        max_delta_path_cost, L, U = __update_trees(
            label[lowest_order_node],
            len(topological_order),
            L,
            U,
            min_path_predecessors,
            max_path_predecessors,
            topological_order,
            costs,
            bush_flows,
            bush_in_links,
        )
        # print(f'max path delta in mainline: {max_delta_path_cost}')
    number_of_edges = len(links_in_bush)
    links_in_bush = __remove_unused_edges(
        links_in_bush=links_in_bush,
        bush_flows=bush_flows,
        to_node=to_node,
        from_node=from_node,
        bush_out_links=bust_out_links,
        bush_in_links=bush_in_links,
        min_path_predecessors=min_path_predecessors,
        tot_links=tot_links,
    )
    if len(links_in_bush) > number_of_edges:
        # print('time for new labels, edges have been removed!')
        max_delta_path_cost, L, U = __update_trees(
            1,
            len(topological_order),
            L,
            U,
            min_path_predecessors,
            max_path_predecessors,
            topological_order,
            costs,
            bush_flows,
            bush_in_links,
        )
    for i in topological_order:
        assert i in L

    return (
        flows,
        bush_flows,
        links_in_bush,
        converged_without_shifts,
        L,
        U,
        bust_out_links,
        bush_in_links,
    )


@njit
def __update_path_flow(
    delta_f,
    start_node,
    end_node,
    predecessor_dict,
    bush_flow,
    out_links,
    derivatives,
    costs,
    capacities,
    ff_tts,
    flows,
):
    new_path_flow = 100000
    new_path_cost = 0
    new_path_derivative = 0
    i = end_node
    while i != start_node:
        (i, j) = (predecessor_dict[i], i)
        link_id = get_link_id(i, j, out_links)
        bush_flow[link_id] = bush_flow[link_id] + delta_f
        flows[link_id] = flows[link_id] + delta_f
        new_path_flow = min(new_path_flow, bush_flow[link_id])
        costs[link_id] = __bpr_cost_single(
            capacity=capacities[link_id],
            ff_tt=ff_tts[link_id],
            flow=flows[link_id],
        )
        derivatives[link_id] = __bpr_derivative_single(
            capacity=capacities[link_id],
            ff_tt=ff_tts[link_id],
            flow=flows[link_id],
        )
        new_path_cost += costs[link_id]
        new_path_derivative += derivatives[link_id]
    return new_path_flow, new_path_cost, new_path_derivative


@njit
def __get_delta_flow_and_cost(
    min_path_flow,
    max_path_flow,
    min_path_cost,
    max_path_cost,
    min_path_derivative,
    max_path_derivative,
):
    if min_path_cost < max_path_cost:
        delta_f = max_path_flow
    else:
        delta_f = -min_path_flow
    assert min_path_flow >= 0
    if (max_path_derivative + min_path_derivative) <= 0:
        if min_path_cost < max_path_cost:
            delta_f = max_path_flow
        else:
            delta_f = -min_path_flow
    else:
        if delta_f >= 0:
            delta_f = min(
                delta_f,
                (max_path_cost - min_path_cost)
                / (min_path_derivative + max_path_derivative),
            )
        else:
            delta_f = max(
                delta_f,
                (max_path_cost - min_path_cost)
                / (min_path_derivative + max_path_derivative),
            )
    return delta_f, max_path_cost - min_path_cost


@njit
def __equalize_cost(
    start_node,
    end_node,
    max_path_flow,
    min_path_flow,
    max_path_cost,
    min_path_cost,
    min_path_derivative,
    max_path_derivative,
    min_path_predecessors,
    max_path_predecessors,
    bush_flow,
    out_links,
    derivatives,
    costs,
    capacities,
    ff_tts,
    flows,
):
    assert start_node != end_node
    total = min_path_flow + max_path_flow
    # print('got into eq cost')
    delta_f, delta_cost = __get_delta_flow_and_cost(
        min_path_flow,
        max_path_flow,
        min_path_cost,
        max_path_cost,
        min_path_derivative,
        max_path_derivative,
    )
    # print(f'delta cost is {delta_cost} with a shift of {delta_f}')
    assert abs(delta_f) < 100000
    while abs(delta_cost) > epsilon / 10 and abs(delta_f) > 0:
        #   print(f'delta cost is {delta_cost}')
        min_path_flow, min_path_cost, min_path_derivative = __update_path_flow(
            delta_f,
            start_node,
            end_node,
            min_path_predecessors,
            bush_flow,
            out_links,
            derivatives,
            costs,
            capacities,
            ff_tts,
            flows,
        )
        #  print('got out of update p flow')
        max_path_flow, max_path_cost, max_path_derivative = __update_path_flow(
            -delta_f,
            start_node,
            end_node,
            max_path_predecessors,
            bush_flow,
            out_links,
            derivatives,
            costs,
            capacities,
            ff_tts,
            flows,
        )
        assert total - (min_path_flow + max_path_flow) < epsilon
        # print('updated path flows')
        delta_f, delta_cost = __get_delta_flow_and_cost(
            min_path_flow,
            max_path_flow,
            min_path_cost,
            max_path_cost,
            min_path_derivative,
            max_path_derivative,
        )
        # print(f'next shift is {delta_f} with cost dif {delta_cost}')
    # print(f'remaining cost differences after eq cost are {delta_cost}')


@njit
def __update_trees(
    k,
    n,
    L,
    U,
    min_path_predecessors,
    max_path_predecessors,
    topological_order,
    costs,
    bush_flows,
    bush_in_links,
):
    """
    k
    """
    assert k >= 0
    assert k <= len(topological_order) - 1
    assert n > 0
    assert n <= len(topological_order)
    max_delta_path_costs = 0
    if k == 0:
        U[0] = 0.0
        L[0] = 0.0
        k = 1
    for j in topological_order[k:n]:
        max_path_predecessors[j], min_path_predecessors[j] = 0, 0
        L[j], U[j] = 100000.0, -100000.0
        for i, link_id in bush_in_links[j]:
            if i not in L:
                print("topological order broken for node i " + str(i))
                print("supposed to be BEFORE node j " + str(j))
                print(L)
            # assert i in L
            # assert j in L
            # assert i in U
            # assert j in U
            # these asserts basically verify whether
            # the topological order is still intact
            if L[i] + costs[link_id] < L[j]:
                L[j] = L[i] + costs[link_id]
                min_path_predecessors[j] = i
            if bush_flows[link_id] > 0 and U[i] + costs[link_id] > U[j]:
                U[j] = U[i] + costs[link_id]
                max_path_predecessors[j] = i
        if max_path_predecessors[j] != 0:
            max_delta_path_costs = max(max_delta_path_costs, U[j] - L[j])
            assert max_delta_path_costs < 99999
        if U[j] > 0:
            assert L[j] <= U[j]
    return max_delta_path_costs, L, U


@njit
def __get_branch_nodes(
    destination,
    min_path_predecessors,
    max_path_predecessors,
    bush_flows,
    out_links,
    costs,
    label,
    derivatives,
):
    """

    Parameters
    ----------
    costs : object
    """
    last_branch_node = destination
    next_min_i = min_path_predecessors[destination]
    next_max_i = max_path_predecessors[destination]
    while next_min_i == next_max_i:
        if next_min_i == next_max_i:
            last_branch_node = next_max_i
        next_min_i = min_path_predecessors[next_min_i]
        next_max_i = max_path_predecessors[next_max_i]

    # print(f'first divergence node found {next_max_i}')
    next_min_link = get_link_id(
        min_path_predecessors[last_branch_node], last_branch_node, out_links
    )
    next_max_link = get_link_id(
        max_path_predecessors[last_branch_node], last_branch_node, out_links
    )
    edges_on_max_path, edges_on_min_path = 1, 1
    min_path_flow, max_path_flow = (
        bush_flows[next_min_link],
        bush_flows[next_max_link],
    )
    min_path_cost, max_path_cost = (costs[next_min_link], costs[next_max_link])
    min_path_derivative, max_path_derivative = 0, 0
    while next_min_i != next_max_i:
        # print(f'the current min label is {label[next_min_i]}with node {next_min_i}')
        # print(f'the current max label is {label[next_max_i]}with node {next_max_i}')
        while label[next_min_i] < label[next_max_i]:
            #   print(f'following max, label is {label[next_max_i]}')
            j = next_max_i
            next_max_i = max_path_predecessors[next_max_i]
            link_id = get_link_id(next_max_i, j, out_links)
            max_path_flow = min(max_path_flow, bush_flows[link_id])
            max_path_cost += costs[link_id]

            max_path_derivative += derivatives[link_id]
            edges_on_max_path += 1
        while label[next_min_i] > label[next_max_i]:
            # print(f'following min, label is {label[next_min_i]}')
            j = next_min_i
            next_min_i = min_path_predecessors[next_min_i]
            link_id = get_link_id(next_min_i, j, out_links)
            min_path_flow = min(min_path_flow, bush_flows[link_id])
            min_path_cost += costs[link_id]
            # min_path, costs are now {min_path_cost}  ')
            min_path_derivative += derivatives[link_id]
            edges_on_min_path += 1
    first_branch_node = next_min_i
    return (
        first_branch_node,
        last_branch_node,
        min_path_flow,
        max_path_flow,
        min_path_cost,
        max_path_cost,
        min_path_derivative,
        max_path_derivative,
    )


@njit
def __shift_flow(
    topological_order,
    L,
    U,
    min_path_predecessors,
    max_path_predecessors,
    derivatives,
    costs,
    out_links,
    label,
    bush_flows,
    capacities,
    flows,
    ff_tts,
    bush_in_links,
):
    lowest_order_node = len(topological_order) - 1
    # print('new run in shift flow')
    for j in topological_order[::-1]:
        if U[j] - L[j] > epsilon / 10:
            # print(f'require shift for destination j {j} with label
            # {label[j]}, cost differences are: {U[j] - L[j]}')
            (
                start_node,
                end_node,
                min_path_flow,
                max_path_flow,
                min_path_cost,
                max_path_cost,
                min_path_derivative,
                max_path_derivative,
            ) = __get_branch_nodes(
                j,
                min_path_predecessors,
                max_path_predecessors,
                bush_flows,
                out_links,
                costs,
                label,
                derivatives,
            )
            total_flow = min_path_flow + max_path_flow
            # print( f'the branch nodes are {start_node, end_node}
            # with labels{label[start_node], label[end_node]},
            # ost dif are{max_path_cost - min_path_cost}')
            if abs(max_path_cost - min_path_cost) > epsilon / 10:
                __equalize_cost(
                    start_node,
                    end_node,
                    max_path_flow,
                    min_path_flow,
                    max_path_cost,
                    min_path_cost,
                    min_path_derivative,
                    max_path_derivative,
                    min_path_predecessors,
                    max_path_predecessors,
                    bush_flows,
                    out_links,
                    derivatives,
                    costs,
                    capacities,
                    ff_tts,
                    flows,
                )
                assert total_flow == min_path_flow + max_path_flow
                # print(f'updating tree between {start_node}
                # and {j} with labels {label[start_node], label[j]}')
                __update_trees(
                    label[start_node],
                    label[j] + 1,
                    L,
                    U,
                    min_path_predecessors,
                    max_path_predecessors,
                    topological_order,
                    costs,
                    bush_flows,
                    bush_in_links,
                )
                # print(f'cost difs are now {U[j] - L[j]}')
                assert abs(U[j] - L[j]) < 99999
            else:
                continue
            lowest_order_node = j

    return lowest_order_node


@njit
def __remove_unused_edges(
    links_in_bush,
    bush_flows,
    bush_out_links,
    bush_in_links,
    min_path_predecessors,
    from_node,
    to_node,
    tot_links,
):
    to_be_removed = np.full(tot_links, False)
    for link, in_bush in enumerate(links_in_bush):
        if in_bush:
            if bush_flows[link] < epsilon / 10:
                to_be_removed[link] = True
    offset = 0
    pruning_counter = 0
    for link, remove in enumerate(to_be_removed):
        if remove:
            i = from_node[link]
            j = to_node[link]
            # print(f'edge under consideration ij: {i,j}')
            if (
                min_path_predecessors[j] != i
            ):  # otherwise the edge is needed for connectivity
                #    print(f'edge {(i,j)} with flow
                #    {bush_flows[edge_map[(i,j)]]} removed ')
                links_in_bush[link] = False
                if bush_out_links[i].size == 2:
                    bush_out_links[i] = np.empty((0, 2), dtype=np.int64)
                else:
                    bush_out_links[i] = bush_out_links[i][bush_out_links[i][:, 0] != j]
                if bush_in_links[j].size == 2:
                    bush_in_links[j] = np.empty((0, 2), dtype=np.int64)
                else:
                    bush_in_links[j] = bush_in_links[j][bush_in_links[j][:, 0] != i]
                pruning_counter += 1
                offset += 1
    # print(f'there are {len(bush_edges)} edges
    # left after pruning the bush by {pruning_counter}')
    return links_in_bush
