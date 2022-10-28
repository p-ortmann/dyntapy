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

from dyntapy.settings import parameters, debugging
from dyntapy.sta.utilities import __bpr_cost_single_toll, __bpr_derivative_single
from dyntapy.graph_utils import _get_link_id

link_cost_function = __bpr_cost_single_toll

# sub modules for Dial's Algorithm B.

epsilon = parameters.static_assignment.dial_b_cost_differences
epsilon_2 = epsilon / 20  # Epsilon that is used on an alternatives basis, replaces


# the expansion factor in Dial's paper.
# needs to be lower than epsilon to achieve an epsilon compliant gap across all
# destinations.


@njit
def __equilibrate_bush(
        costs,
        bush_flows,
        destination,
        flows,
        topological_order,
        derivatives,
        turns_in_bush,
        capacities,
        ff_tts,
        bush_out_turns,
        bush_in_turns,
        epsilon,
        global_out_turns,
        tot_turns,
        tot_links,
        to_links,
        from_links,
        tolls
):
    # we equilibrate each bush to convergence before moving on to the next ..
    # once shifting in the bush has reached equilibrium
    # we try to add shortcut links from the graph
    # and start shifting again, until this doesn't yield improvements anymore.
    converged_without_shifts = False
    # max_path_predecessors = Dict()
    min_path_successor = np.full(tot_links, -1)
    # min_path_predecessors = Dict()
    max_path_successor = np.full(tot_links, -1)
    max_path_successor[destination] = -1
    min_path_successor[destination] = -1
    L = np.full(tot_links, np.inf)
    U = np.full(tot_links, -np.inf)
    label = np.empty(tot_links, np.uint32)
    U[destination] = 0.0
    L[destination] = 0.0
    for index, link in enumerate(topological_order):
        label[link] = index
    max_delta_path_cost, L, U = __update_trees(
        1,
        len(topological_order),
        L,
        U,
        min_path_successor,
        max_path_successor,
        topological_order,
        costs,
        bush_flows,
        bush_out_turns,
    )

    if debugging:
        print(
            f"________the remaining cost differences in this bush for origin "
            f"{destination} "
            f"are {float(max_delta_path_cost)}______"
        )
    for i in topological_order:
        assert L[i] != np.inf

    if epsilon > max_delta_path_cost:
        converged_without_shifts = True

        if debugging:
            print(
                f"no shifts were ever necessary, delta: {max_delta_path_cost} smaller "
                f"than epsilon {epsilon}"
            )
    while epsilon < max_delta_path_cost:
        if debugging:
            print(
                f"calling shift flow, cost differences are:{max_delta_path_cost} "
                f"larger "
                f"than {epsilon} "
            )
        lowest_order_node = __shift_flow(
            topological_order,
            L,
            U,
            min_path_successor,
            max_path_successor,
            derivatives,
            costs,
            global_out_turns,
            label,
            bush_flows,
            capacities,
            flows,
            ff_tts,
            bush_out_turns,
            tolls
        )
        if debugging:
            print(f"updating trees, branch node is: {lowest_order_node}")
        max_delta_path_cost, L, U = __update_trees(
            label[lowest_order_node],
            len(topological_order),
            L,
            U,
            min_path_successor,
            max_path_successor,
            topological_order,
            costs,
            bush_flows,
            bush_out_turns,
        )
        if debugging:
            print(f"max path delta in mainline: {max_delta_path_cost}")
    number_of_turns = np.sum(turns_in_bush)
    turns_in_bush = __remove_unused_turns(
        turns_in_bush=turns_in_bush,
        bush_flows=bush_flows,
        to_link=to_links,
        from_link=from_links,
        bush_out_turns=bush_out_turns,
        bush_in_turns=bush_in_turns,
        min_path_successor=min_path_successor,
        tot_turns=tot_turns,
    )
    if np.sum(turns_in_bush) < number_of_turns:
        if debugging:
            print("time for new labels, edges have been removed!")
        max_delta_path_cost, L, U = __update_trees(
            1,
            len(topological_order),
            L,
            U,
            min_path_successor,
            max_path_successor,
            topological_order,
            costs,
            bush_flows,
            bush_out_turns,
        )
    for i in topological_order:
        assert L[i] != np.inf

    return (
        flows,
        bush_flows,
        turns_in_bush,
        converged_without_shifts,
        L,
        U,
        bush_out_turns,
        bush_in_turns,
    )


@njit
def __update_path_flow(
        delta_f,
        start_link,
        end_link,
        successor,
        bush_flow,
        out_turns,
        derivatives,
        costs,
        capacities,
        ff_tts,
        flows,
        tolls
):
    new_path_flow = 100000
    new_path_cost = 0
    new_path_derivative = 0
    i = start_link
    while i != end_link:
        (i, j) = (i, successor[i])
        turn_id = _get_link_id(i, j, out_turns)
        if not bush_flow[turn_id] + delta_f >= 0:
            print(f'error turn {turn_id} links {i} and {j}')
            raise AssertionError
        bush_flow[turn_id] = bush_flow[turn_id] + delta_f
        flows[j] = flows[j] + delta_f
        new_path_flow = min(new_path_flow, bush_flow[turn_id])
        # get link flows
        costs[turn_id] = link_cost_function(
            capacity=capacities[i],
            ff_tt=ff_tts[i],
            flow=flows[i], toll=tolls[i]
        )
        # get link derivatives
        derivatives[turn_id] = __bpr_derivative_single(
            capacity=capacities[j],
            ff_tt=ff_tts[j],
            flow=flows[j],
        )
        new_path_cost += costs[turn_id]
        new_path_derivative += derivatives[turn_id]
        i = successor[i]
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
    elif min_path_cost> max_path_cost:
        delta_f = -min_path_flow
    else:
        # equal
        return 0,0
    assert min_path_flow >= 0
    if (max_path_derivative + min_path_derivative) <= 0:
        if min_path_cost < max_path_cost:
            delta_f = max_path_flow
        elif min_path_cost> max_path_cost:
            delta_f = -min_path_flow
        else:
            # equal
            return 0,0
    else:
        if delta_f > 0:
            delta_f = min(
                delta_f,
                (max_path_cost - min_path_cost)
                / (min_path_derivative + max_path_derivative),
            )

            assert delta_f > 0
        elif delta_f < 0:
            delta_f = max(
                delta_f,
                (max_path_cost - min_path_cost)
                / (min_path_derivative + max_path_derivative),
            )
            if not delta_f < 0:
                print('error')
                print(min_path_cost)
                print(max_path_cost)
                print(min_path_derivative)
                print(max_path_derivative)
                print(delta_f)
                raise AssertionError
    return delta_f, max_path_cost - min_path_cost


@njit
def __equalize_cost(
        start_link,
        end_link,
        max_path_flow,
        min_path_flow,
        max_path_cost,
        min_path_cost,
        min_path_derivative,
        max_path_derivative,
        min_path_successors,
        max_path_successors,
        bush_flow,
        out_turns,
        derivatives,
        costs,
        capacities,
        ff_tts,
        flows,
        tolls
):
    assert start_link != end_link
    assert min_path_flow >= 0
    assert max_path_flow >= 0
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
    while abs(delta_cost) > epsilon_2 and abs(delta_f) > 0:
        #   print(f'delta cost is {delta_cost}')
        min_path_flow, min_path_cost, min_path_derivative = __update_path_flow(
            delta_f,
            start_link,
            end_link,
            min_path_successors,
            bush_flow,
            out_turns,
            derivatives,
            costs,
            capacities,
            ff_tts,
            flows,
            tolls
        )
        #  print('got out of update p flow')
        max_path_flow, max_path_cost, max_path_derivative = __update_path_flow(
            -delta_f,
            start_link,
            end_link,
            max_path_successors,
            bush_flow,
            out_turns,
            derivatives,
            costs,
            capacities,
            ff_tts,
            flows,
            tolls
        )
        assert (
                np.abs(total - (min_path_flow + max_path_flow)) < np.finfo(
            np.float32).eps
        )
        # bush flow
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
        min_path_successor,
        max_path_successor,
        topological_order,
        costs,
        bush_flows,
        bush_out_turns,
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
        U[topological_order[0]] = 0.0
        L[topological_order[0]] = 0.0
        k = 1
    for i in topological_order[k:n]:
        L[i], U[i] = np.inf, -np.inf
        max_path_successor[i] = -1
        min_path_successor[i] = -1
        for j, turn_id in bush_out_turns[i]:
            if L[j] == np.inf:
                raise AssertionError
                if debugging:
                    print("topological order broken for node i " + str(i))
                    print("supposed to be BEFORE node j " + str(j))
                    print(L)

            # assert i in L
            # assert j in L
            # assert i in U
            # assert j in U
            # these assert statements verify whether
            # the topological order is still intact
            if L[i] > L[j] + costs[turn_id]:
                L[i] = L[j] + costs[turn_id]
                min_path_successor[i] = j
            if bush_flows[turn_id] > np.finfo(np.float32).eps and U[i] < U[j] + costs[
                turn_id]:
                U[i] = U[j] + costs[turn_id]
                max_path_successor[i] = j
        if max_path_successor[i] != -1:
            max_delta_path_costs = max(max_delta_path_costs, U[i] - L[i])
            if max_delta_path_costs>9999999:
                print('hi')
            assert max_delta_path_costs < 9999999
        if U[i] > 0:
            assert L[i] <= U[i]
    assert np.all(L[topological_order] != np.inf)
    return max_delta_path_costs, L, U


@njit
def __get_branch_links(
        origin,
        min_path_successor,
        max_path_successor,
        bush_flows,
        out_turns,
        turn_costs,
        label,
        turn_derivatives,  # turn derivatives
):
    """

    Parameters
    ----------
    costs : object
    """
    first_branch_link = origin
    next_min_i = min_path_successor[origin]
    next_max_i = max_path_successor[origin]
    max_turns = List()
    min_turns = List()
    while next_min_i == next_max_i:
        if next_min_i == next_max_i:
            first_branch_link = next_max_i
        next_min_i = min_path_successor[next_min_i]
        next_max_i = max_path_successor[next_max_i]

    # print(f'first divergence node found {next_max_i}')
    next_min_turn = _get_link_id(
        first_branch_link, min_path_successor[first_branch_link], out_turns
    )
    next_max_turn = _get_link_id(
        first_branch_link, max_path_successor[first_branch_link], out_turns
    )
    max_turns.append(next_max_turn)
    min_turns.append(next_min_turn)
    turns_on_max_path, turns_on_min_path = 1, 1
    min_path_flow, max_path_flow = (
        bush_flows[next_min_turn],
        bush_flows[next_max_turn],
    )
    min_path_cost, max_path_cost = (
        max(turn_costs[next_min_turn] - turn_costs[next_max_turn], 0),
        max(turn_costs[next_max_turn] - turn_costs[next_min_turn], 0))
    # omitting the shared cost of the origin link here.
    min_path_derivative, max_path_derivative = turn_derivatives[next_min_turn], \
                                               turn_derivatives[next_max_turn]
    while next_min_i != next_max_i:
        # print(f'the current min label is {label[next_min_i]}with node {next_min_i}')
        # print(f'the current max label is {label[next_max_i]}with node {next_max_i}')
        last_min_path_derivative = -1
        last_max_path_derivative = -1
        while label[next_min_i] < label[next_max_i]:
            #   print(f'following max, label is {label[next_max_i]}')
            j = next_max_i
            next_max_i = max_path_successor[next_max_i]
            turn_id = _get_link_id(j, next_max_i, out_turns)
            max_turns.append(turn_id)
            max_path_flow = min(max_path_flow, bush_flows[turn_id])
            max_path_cost += turn_costs[turn_id]
            last_max_path_derivative = turn_derivatives[turn_id]
            max_path_derivative += turn_derivatives[turn_id]
            turns_on_max_path += 1
        while label[next_min_i] > label[next_max_i]:
            # print(f'following min, label is {label[next_min_i]}')
            j = next_min_i
            next_min_i = min_path_successor[next_min_i]
            turn_id = _get_link_id(j, next_min_i, out_turns)
            min_turns.append(turn_id)
            min_path_flow = min(min_path_flow, bush_flows[turn_id])
            min_path_cost += turn_costs[turn_id]
            # min_path, costs are now {min_path_cost}  ')
            last_min_path_derivative = turn_derivatives[turn_id]
            min_path_derivative += turn_derivatives[turn_id]
            turns_on_min_path += 1
    max_path_derivative = max_path_derivative - last_max_path_derivative
    min_path_derivative = min_path_derivative - last_min_path_derivative
    last_branch_link = next_min_i
    assert min_path_flow >= 0
    assert max_path_flow >= 0
    for turn in max_turns:
        if bush_flows[turn] < max_path_flow:
            print(f'error with turn {turn}')

    for turn in min_turns:
        if bush_flows[turn] < min_path_flow:
            print(f'error with turn {turn}')
    return (
        first_branch_link,
        last_branch_link,
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
        min_path_successors,
        max_path_successors,
        derivatives,
        costs,
        out_turns,
        label,
        bush_flows,
        capacities,
        flows,
        ff_tts,
        bush_out_turns,
        tolls
):
    lowest_order_link = 1
    # print('new run in shift flow')
    for j in topological_order[::-1]:
        if U[j] - L[j] > epsilon_2:  # the shifts here need to be tighter so that
            # the overall gap goes below the threshold
            # print(f'require shift for destination j {j} with label {label[j]}, '
            #       f'cost differences are: {U[j] - L[j]}')
            (
                start_link,
                end_link,
                min_path_flow,
                max_path_flow,
                min_path_cost,
                max_path_cost,
                min_path_derivative,
                max_path_derivative,
            ) = __get_branch_links(
                j,
                min_path_successors,
                max_path_successors,
                bush_flows,
                out_turns,
                costs,
                label,
                derivatives,
            )
            total_flow = min_path_flow + max_path_flow
            # print(f'the branch nodes are '
            #       f'{start_node, end_node} with label'
            #       f's{label[start_node], label[end_node]}, cost dif ar'
            #       f'e{max_path_cost - min_path_cost}')
            if abs(max_path_cost - min_path_cost) > 0:
                __equalize_cost(
                    start_link,
                    end_link,
                    max_path_flow,
                    min_path_flow,
                    max_path_cost,
                    min_path_cost,
                    min_path_derivative,
                    max_path_derivative,
                    min_path_successors,
                    max_path_successors,
                    bush_flows,
                    out_turns,
                    derivatives,
                    costs,
                    capacities,
                    ff_tts,
                    flows,
                    tolls
                )
                assert total_flow == min_path_flow + max_path_flow
                # print(f'updating tree between {end_link} and {j} with labels'
                #       f' {label[end_link], label[j]}')
                __update_trees(
                    label[end_link],
                    label[j] + 1,
                    L,
                    U,
                    min_path_successors,
                    max_path_successors,
                    topological_order,
                    costs,
                    bush_flows,
                    bush_out_turns,
                )
                # print(f'cost differences now {U[j] - L[j]}')
                assert abs(U[j] - L[j]) < 99999
            else:
                continue
            lowest_order_link = j

    return lowest_order_link


@njit
def __remove_unused_turns(
        turns_in_bush,
        bush_flows,
        bush_out_turns,
        bush_in_turns,
        min_path_successor,
        from_link,
        to_link,
        tot_turns,
):
    if debugging:
        print("removing edges called")
    to_be_removed = np.full(tot_turns, False)
    for turn, in_bush in enumerate(turns_in_bush):
        if in_bush:
            if bush_flows[turn] < np.finfo(np.float32).eps:
                to_be_removed[turn] = True
    offset = 0
    pruning_counter = 0
    for turn, remove in enumerate(to_be_removed):
        if remove:
            i = from_link[turn]
            j = to_link[turn]
            if debugging:
                pass
                # print(f"edge under consideration ij: {i, j}")
            try:
                if (
                        len(bush_out_turns[i]) > 1 and min_path_successor[i] != j
                ):  # otherwise the edge is needed for connectivity
                    #    print(f'edge {(i,j)} with flow
                    #    {bush_flows[edge_map[(i,j)]]} removed ')
                    turns_in_bush[turn] = False
                    if bush_out_turns[i].size == 2:
                        bush_out_turns[i] = np.empty((0, 2), dtype=np.int64)
                    else:
                        bush_out_turns[i] = bush_out_turns[i][
                            bush_out_turns[i][:, 0] != j
                            ]
                    if bush_in_turns[j].size == 2:
                        bush_in_turns[j] = np.empty((0, 2), dtype=np.int64)
                    else:
                        bush_in_turns[j] = bush_in_turns[j][bush_in_turns[j][:, 0] != i]
                    pruning_counter += 1
                    offset += 1

                    if debugging:
                        print(f"removed edge ij: {i, j}, turn {turn}")
            except Exception:
                raise Exception

    # print(f'there are {len(bush_edges)} edges
    # left after pruning the bush by {pruning_counter}')
    return turns_in_bush
