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

from heapq import heappop, heappush
from dyntapy.csr import BCSRMatrix, UI32CSRMatrix
from dyntapy.sta.utilities import _bpr_cost_single_toll, _bpr_derivative_single
from dyntapy.graph_utils import _get_link_id, dijkstra_all
from dyntapy.settings import debugging, debugging_full

link_cost_function = _bpr_cost_single_toll

# sub modules for Dial's Algorithm B.

# the expansion factor in Dial's paper.
# needs to be lower than epsilon to achieve an epsilon compliant gap across all
# destinations.


@njit
def _remove_unused_turns(
    L,
    out_turns,
    bush_flows,
    bush_out_turns,
    min_path_successors,
    to_link,
):
    if debugging:
        print("removing turns called")
    for i in range(bush_out_turns.tot_rows):
        for idx, turn in enumerate(bush_out_turns.get_nnz(i)):
            j = out_turns.get_row(i)[idx]
            if bush_flows[turn] > np.finfo(np.float32).eps:
                if debugging_full:
                    if L[j] > L[i]:
                        # occurs in initial iterations near the origin
                        print("retaining inefficient turn")
                        print(turn)
                        print(L[j] - L[i])
            if (
                bush_out_turns.get_row(i)[idx]
                and bush_flows[turn] < np.finfo(np.float32).eps
            ):
                if (
                    bush_out_turns.get_row(i).sum() > 1
                    and min_path_successors[i] != to_link[turn]
                ):
                    bush_out_turns.get_row(i)[idx] = False
                    if debugging_full:
                        print(f"removing turn {turn} from link {i}")


@njit
def topological_sort(
    out_turns,
    in_turns,
    bush_out_turns,
    tot_reachable_nodes,
    tot_nodes,
    destination_link,
    old_topological_order,
):
    # topological sort on a graph assuming that the defined bush_outturns form a
    # a connected DAG, (Directed Acyclic Graph)
    # destinations forms the root as sink
    order = np.zeros(tot_reachable_nodes, dtype=np.uint32)
    order[0] = destination_link
    my_heap = []
    my_heap.append((np.uint32(0), np.uint32(destination_link)))
    visited_node = np.full(tot_nodes, False)
    idx = 0
    while my_heap:
        my_tuple = heappop(my_heap)
        pos = my_tuple[0]
        j = my_tuple[1]
        assert not visited_node[j]
        visited_node[j] = True
        order[idx] = j
        idx = idx + 1
        pos_count = pos
        for i in in_turns.get_row(j):
            visited_all_nodes_in_forward_star = False
            if j in out_turns.get_row(i)[bush_out_turns.get_row(i)]:
                visited_all_nodes_in_forward_star = True  # have all outgoing links
                # from i been processed?
                for j2 in out_turns.get_row(i)[bush_out_turns.get_row(i)]:
                    if not visited_node[j2]:
                        visited_all_nodes_in_forward_star = False
            if visited_all_nodes_in_forward_star:
                pos_count += 1
                heappush(my_heap, (np.uint32(pos_count), np.uint32(i)))
    if not len(set(order)) == tot_reachable_nodes:  # if this fails forward and
        if debugging:
            for i in old_topological_order:
                if i not in set(order):
                    print(i)
            print("_____")
            print(len(set(order)))
            print(tot_reachable_nodes)
        raise AssertionError
    # backward stars do
    # not form DAG
    return order


@njit
def _update_bush(
    U,
    L,
    turn_costs,
    bush_out_turns: BCSRMatrix,
    bush_flows,
    destination,
    global_in_turns,
    global_out_turns,
    tot_reachable_links,
    old_topological_order,
    eps=np.finfo(np.float32).eps,
):
    # if eps is set too close to 0, cycles may be created.
    # All solutions to the bush-based algorithms may retain inefficient edges,
    # within the PAS convergence bound that is set.
    # if a topological order cannot be generated turns are removed in order of
    # efficiency until a topological order is found.
    added_turns = List()
    added_turns.append((0, 0))
    deltas = List()
    deltas.append(0.0)
    deltas.pop()
    added_turns.pop()
    new_turns_added = False
    tot_links = global_out_turns.tot_rows
    for i in old_topological_order:
        for idx, (j, turn_id) in enumerate(
            zip(
                global_out_turns.get_row(i),
                global_out_turns.get_nnz(i),
            )
        ):
            is_contained_turn = bush_out_turns.get_row(i)[idx]
            if not is_contained_turn:
                if L[i] - eps > turn_costs[turn_id] + L[j]:
                    delta = L[i] - turn_costs[turn_id] - L[j]
                    deltas.append(delta)
                    if debugging_full:
                        print(f"adding turn {turn_id} from link {i} to link {j}")
                    new_turns_added = True
                    added_turns.append((i, idx))
                    bush_out_turns.get_row(i)[idx] = True
                    if debugging:
                        try:
                            topological_sort(
                                global_out_turns,
                                global_in_turns,
                                bush_out_turns,
                                len(old_topological_order),
                                tot_links,
                                destination,
                                old_topological_order,
                            )
                        except:
                            print("topological order compromised by adding turn")
                            print(turn_id)
                            # the way it is set up right now this is expected
                            # behaviour, the fallback below handles it.
                            # on real networks it is quite rare that cycles are created
                            # through this process.
                            # The below is essentially an escape hatch IF it does
                            # happen.
                            # Aggressively adding turns, like above, can speed up the
                            # equilibrium finding though.

    try:
        new_topological_order = topological_sort(
            global_out_turns,
            global_in_turns,
            bush_out_turns,
            len(old_topological_order),
            tot_links,
            destination,
            old_topological_order,
        )
    except:
        # no topological order with current epsilon, readding turns one by one until
        # graph is no longer cyclic.
        success = False
        if debugging:
            print("fallback to restricted arc insertion to prevent cyclic graph")

        while not success:

            min_delta_idx = 0
            min_delta = 1000000
            for _idx, delta in enumerate(deltas):
                if delta < min_delta:
                    min_delta = delta
                    min_delta_idx = _idx
            min_i = added_turns[min_delta_idx][0]
            bush_out_turns.get_row(min_i)[added_turns[min_delta_idx][1]] = False
            deltas.pop(min_delta_idx)
            added_turns.pop(min_delta_idx)
            if len(added_turns) == 0:
                new_turns_added = False

            try:
                new_topological_order = topological_sort(
                    global_out_turns,
                    global_in_turns,
                    bush_out_turns,
                    len(old_topological_order),
                    tot_links,
                    destination,
                    old_topological_order,
                )
                success = True
            except:
                pass

    return new_turns_added, new_topological_order


@njit
def update_bush_flow(
    delta,
    turn,
    in_turns,
    j,
    capacities,
    ff_tts,
    tolls,
    local_bush_flow,
    global_turn_flows,
    turn_derivatives,
    costs,
):
    local_bush_flow[turn] += delta
    global_turn_flows[turn] += delta
    if not global_turn_flows[turn] - local_bush_flow[turn] > -np.finfo(np.float32).eps:
        print(global_turn_flows[turn] - local_bush_flow[turn])
        print("local and global turn flows inconsistent, unrelated to shift")
        print(f"concerning {turn}")
        print(global_turn_flows[turn])
        print(local_bush_flow[turn])
        print(f"j {j}")
        print(delta)
        print("inturns and inlinks j")
        print(in_turns.get_nnz(j))
        print(in_turns.get_row(j))
        raise AssertionError
    if global_turn_flows[turn] < 0:
        if not global_turn_flows[turn] > -np.finfo(np.float32).eps:
            print(global_turn_flows[turn])
            print(f"continuity error turn {turn}, global")
            raise AssertionError
        global_turn_flows[turn] = 0  # avoiding negative flows due to rounding
    if local_bush_flow[turn] < 0:
        if not local_bush_flow[turn] > -np.finfo(np.float32).eps:
            print(local_bush_flow[turn])
            print(f"continuity error turn {turn}, local")
            raise AssertionError
        local_bush_flow[turn] = 0  # avoiding negative flows due to rounding
    link_flow = global_turn_flows[in_turns.get_nnz(j)].sum()
    updated_turn_cost = link_cost_function(
        capacity=capacities[j], ff_tt=ff_tts[j], flow=link_flow, toll=tolls[j]
    )
    updated_turn_derivative = _bpr_derivative_single(
        capacity=capacities[j],
        ff_tt=ff_tts[j],
        flow=link_flow,
    )
    for turn in in_turns.get_nnz(j):
        costs[turn] = updated_turn_cost
        turn_derivatives[turn] = updated_turn_derivative


@njit
def find_reachable_topo_links(
    bush_out_turns: BCSRMatrix, out_turns, origins, tot_links
):
    reachable = np.full(tot_links, False)
    for origin in origins:
        to_process = List()
        to_process.append(origin)
        while len(to_process) > 0:
            i = to_process.pop()
            if not reachable[i]:
                reachable[i] = True
            else:
                continue
            for j in out_turns.get_row(i)[bush_out_turns.get_row(i)]:
                to_process.append(j)
    # print(f"origin is {origin}")
    # print(np.sum(reachable))
    return reachable


@njit
def _equilibrate_bush(
    costs,
    bush_flows,
    turn_flows,
    destination,
    origins,
    topological_order,
    derivatives,
    capacities,
    ff_tts,
    bush_out_turns,
    epsilon,
    global_out_turns: UI32CSRMatrix,
    global_in_turns,
    to_links,
    tolls,
):
    pas_epsilon = (
        epsilon / 30
    )  # Epsilon that is used on an alternatives basis, replaces
    tot_links = global_out_turns.tot_rows
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
    L = np.full(tot_links, np.inf, dtype=np.float64)
    U = np.full(tot_links, -np.inf, dtype=np.float64)
    label = np.empty(tot_links, np.uint32)
    for index, link in enumerate(topological_order):
        label[link] = index
    U[destination] = 0.0
    L[destination] = 0.0

    # for a set of active origins the below will find the most
    # topologically upstream one among them and
    # constructs a set of active links by identifying all topologically downstream
    # links that are
    # loaded, from each of those links it searches forward to find the set of links
    # that are reachable..
    link_flows = np.zeros(tot_links)
    max_idx = np.max(label[origins])
    active_links = np.full(tot_links, False)
    for link in range(tot_links):
        link_flows[link] = max(
            bush_flows[global_out_turns.get_nnz(link)].sum(),
            bush_flows[global_in_turns.get_nnz(link)].sum(),
        )
        if label[link] < max_idx and link_flows[link] > 0:
            active_links[link] = True
    reachable = find_reachable_topo_links(
        bush_out_turns, global_out_turns, np.argwhere(active_links).flatten(), tot_links
    )
    tot_reachable = np.sum(reachable)
    reduced_topological_order = np.empty(tot_reachable, dtype=np.uint32)
    idx = 0
    for link in topological_order:
        if reachable[link]:
            reduced_topological_order[idx] = link
            idx = idx + 1
    for index, link in enumerate(reduced_topological_order):
        label[link] = index
    max_delta_path_cost, L, U = _update_trees(
        1,
        len(reduced_topological_order),
        L,
        U,
        min_path_successor,
        max_path_successor,
        reduced_topological_order,
        costs,
        bush_flows,
        global_out_turns,
        bush_out_turns,
    )

    if debugging:
        print(
            f"________the remaining cost differences in this bush for origin "
            f"{destination} "
            f"are {float(max_delta_path_cost)}______"
        )
    for i in reduced_topological_order:
        assert L[i] != np.inf

    if epsilon > max_delta_path_cost:
        converged_without_shifts = True

        if debugging:
            print(
                f"no shifts were ever necessary, delta: {max_delta_path_cost} smaller "
                f"than epsilon {epsilon}"
            )
    while max_delta_path_cost > epsilon:
        if debugging:
            print(
                f"calling shift flow, cost differences are:{max_delta_path_cost} "
                f"larger "
                f"than {epsilon} "
            )
        lowest_order_node = _shift_flow(
            reduced_topological_order,
            L,
            U,
            min_path_successor,
            max_path_successor,
            derivatives,
            costs,
            label,
            turn_flows,
            bush_flows,
            capacities,
            ff_tts,
            global_in_turns,
            global_out_turns,
            bush_out_turns,
            tolls,
            pas_epsilon,
        )
        if debugging:
            print(f"updating trees, branch node is: {lowest_order_node}")
        max_delta_path_cost, L, U = _update_trees(
            label[lowest_order_node],
            len(reduced_topological_order),
            L,
            U,
            min_path_successor,
            max_path_successor,
            reduced_topological_order,
            costs,
            bush_flows,
            global_out_turns,
            bush_out_turns,
        )

        # print("max path delta in mainline")
        # print(max_delta_path_cost)
        if debugging:
            print(f"max path delta in mainline: {max_delta_path_cost}")

    max_delta_path_cost, L, U = _update_trees(
        1,
        len(topological_order),
        L,
        U,
        min_path_successor,
        max_path_successor,
        topological_order,
        costs,
        bush_flows,
        global_out_turns,
        bush_out_turns,
    )

    if debugging:
        print(f"max path delta before updating bush: {max_delta_path_cost}")
    _remove_unused_turns(
        L, global_out_turns, bush_flows, bush_out_turns, min_path_successor, to_links
    )
    for i in topological_order:
        if L[i] == np.inf:
            raise AssertionError
    if debugging:
        for i in topological_order:
            if L[i] == np.inf:
                raise AssertionError
    # for origin in origins:
    # print('destination')
    # print(destination)
    # print('origin')
    # print(origin)
    # print('dif')
    # print(U[origin]-L[origin])
    turns_added, new_topological_order = _update_bush(
        U,
        L,
        costs,
        bush_out_turns,
        bush_flows,
        destination,
        global_in_turns,
        global_out_turns,
        len(topological_order),
        topological_order,
        eps=np.finfo(np.float32).eps,
    )
    if debugging:
        if turns_added:
            print("turns added after equilibration, bush not converged")
        else:
            print("no turns added")
        if not turns_added:

            is_centroid = np.full(tot_links, False)
            ssp_dist, succ = dijkstra_all(
                costs, global_in_turns, destination, is_centroid
            )
            for origin in origins:
                dif = L[origin] - ssp_dist[origin]
                # print('from dest')
                # print(destination)
                # print('ORIGIN')
                # print(origin)
                # print('L')
                # print(L[origin])
                # print('U')
                # print(U[origin])
                # print('ssp')
                # print(ssp_dist[origin])
                if dif > np.finfo(np.float32).eps * 10:
                    print(dif)
                    print(origin)
                    print(destination)
                    # verify the path
                    cur_link = origin
                    dist = 0
                    while True:
                        for to_link, turn in zip(
                            global_out_turns.get_row(cur_link),
                            global_out_turns.get_nnz(cur_link),
                        ):
                            if to_link == succ[cur_link]:
                                dist += costs[turn]
                                for turn2, is_included in zip(
                                    bush_out_turns.get_nnz(cur_link),
                                    bush_out_turns.get_row(cur_link),
                                ):
                                    if turn2 == turn:
                                        if not is_included:
                                            print("turn not included")
                                            print(L[cur_link])
                                            print(L[to_link])
                                        else:
                                            print("ssp turn included;")
                        cur_link = succ[cur_link]

                        if cur_link == destination:
                            break
                    raise AssertionError  # the distance reported should be the same
                    # etiher not all shortest path turns are included..
                    # or there is an issue in the cost updating
                    # if this happens it will most likely be related to precision loss..
                    # which is why there is a threshold value for the assertion error
    new_labels = np.full(global_out_turns.tot_rows, global_out_turns.tot_rows + 1)

    for label, link in enumerate(new_topological_order):
        new_labels[link] = label

    return (
        turn_flows,
        bush_flows,
        new_topological_order,
        new_labels,
        converged_without_shifts and not turns_added,
        bush_out_turns,
        min_path_successor,
    )


@njit
def _update_path_flow(
    delta_f,
    start_link,
    end_link,
    successor,
    turn_flows,
    bush_flow,
    global_in_turns,
    global_out_turns,
    derivatives,
    costs,
    capacities,
    ff_tts,
    tolls,
):
    new_path_flow = 100000
    new_path_cost = 0
    new_path_derivative = 0
    i = start_link
    while i != end_link:
        (i, j) = (i, successor[i])
        turn_id = _get_link_id(i, j, global_out_turns)
        if not bush_flow[turn_id] + delta_f >= 0:
            print(f"error turn {turn_id} links {i} and {j}")
            raise AssertionError
        update_bush_flow(
            delta_f,
            turn_id,
            global_in_turns,
            j,
            capacities,
            ff_tts,
            tolls,
            bush_flow,
            turn_flows,
            derivatives,
            costs,
        )
        new_path_flow = min(new_path_flow, bush_flow[turn_id])
        new_path_cost += costs[turn_id]
        new_path_derivative += derivatives[turn_id]
        i = successor[i]
    return new_path_flow, new_path_cost, new_path_derivative


@njit
def _get_delta_flow_and_cost(
    min_path_flow,
    max_path_flow,
    min_path_cost,
    max_path_cost,
    min_path_derivative,
    max_path_derivative,
):
    if min_path_cost < max_path_cost:
        delta_f = max_path_flow
    elif min_path_cost > max_path_cost:
        delta_f = -min_path_flow
    else:
        # equal
        return 0, 0
    assert min_path_flow >= 0
    if (max_path_derivative + min_path_derivative) <= 0:
        if min_path_cost < max_path_cost:
            delta_f = max_path_flow
        elif min_path_cost > max_path_cost:
            delta_f = -min_path_flow
        else:
            # equal
            return 0, 0
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
                print("error")
                print(min_path_cost)
                print(max_path_cost)
                print(min_path_derivative)
                print(max_path_derivative)
                print(delta_f)
                raise AssertionError
    return delta_f, max_path_cost - min_path_cost


@njit
def _equalize_cost(
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
    turn_flows,
    bush_flow,
    global_in_turns,
    global_out_turns,
    derivatives,
    costs,
    capacities,
    ff_tts,
    tolls,
    pas_epsilon,
):
    assert start_link != end_link
    assert min_path_flow >= 0
    assert max_path_flow >= 0
    total = min_path_flow + max_path_flow
    # print('got into eq cost')
    delta_f, delta_cost = _get_delta_flow_and_cost(
        min_path_flow,
        max_path_flow,
        min_path_cost,
        max_path_cost,
        min_path_derivative,
        max_path_derivative,
    )
    # print(f'delta cost is {delta_cost} with a shift of {delta_f}')
    assert abs(delta_f) < 100000
    while abs(delta_cost) > pas_epsilon and abs(delta_f) > 0:
        #   print(f'delta cost is {delta_cost}')
        min_path_flow, min_path_cost, min_path_derivative = _update_path_flow(
            delta_f,
            start_link,
            end_link,
            min_path_successors,
            turn_flows,
            bush_flow,
            global_in_turns,
            global_out_turns,
            derivatives,
            costs,
            capacities,
            ff_tts,
            tolls,
        )
        #  print('got out of update p flow')
        max_path_flow, max_path_cost, max_path_derivative = _update_path_flow(
            -delta_f,
            start_link,
            end_link,
            max_path_successors,
            turn_flows,
            bush_flow,
            global_in_turns,
            global_out_turns,
            derivatives,
            costs,
            capacities,
            ff_tts,
            tolls,
        )
        assert (
            np.abs(total - (min_path_flow + max_path_flow)) < np.finfo(np.float32).eps
        )
        # bush flow
        # print('updated path flows')
        delta_f, delta_cost = _get_delta_flow_and_cost(
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
def _update_trees(
    k,
    n,
    L,
    U,
    min_path_successor,
    max_path_successor,
    topological_order,
    costs,
    bush_flows,
    out_turns,
    bush_out_turns: BCSRMatrix,
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
        U[topological_order[0]] = 0
        L[topological_order[0]] = 0
        k = 1
    for i in topological_order[k:n]:
        L[i], U[i] = np.inf, -np.inf
        max_path_successor[i] = -1
        min_path_successor[i] = -1
        for j, turn_id in zip(
            out_turns.get_row(i)[bush_out_turns.get_row(i)],
            out_turns.get_nnz(i)[bush_out_turns.get_row(i)],
        ):
            if L[j] == np.inf:
                if debugging:
                    print("topological order broken for node i " + str(i))
                    print("supposed to be BEFORE node j " + str(j))
                    print(L)
                raise AssertionError

            # assert i in L
            # assert j in L
            # assert i in U
            # assert j in U
            # these assert statements verify whether
            # the topological order is still intact
            if L[i] > L[j] + costs[turn_id]:
                L[i] = L[j] + costs[turn_id]
                min_path_successor[i] = j
            if (
                bush_flows[turn_id] > np.finfo(np.float32).eps
                and U[i] < U[j] + costs[turn_id]
            ):
                U[i] = U[j] + costs[turn_id]
                max_path_successor[i] = j
        if max_path_successor[i] != -1:
            max_delta_path_costs = max(max_delta_path_costs, U[i] - L[i])
            assert max_delta_path_costs < 9999999
        if U[i] > 0:
            assert L[i] <= U[i]
    assert np.all(L[topological_order] != np.inf)
    return max_delta_path_costs, L, U


@njit
def _get_branch_links(
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
        max(turn_costs[next_max_turn] - turn_costs[next_min_turn], 0),
    )
    # omitting the shared cost of the origin link here.
    min_path_derivative, max_path_derivative = (
        turn_derivatives[next_min_turn],
        turn_derivatives[next_max_turn],
    )
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
    if debugging:
        for turn in max_turns:
            if bush_flows[turn] < max_path_flow:
                print(f"error with turn {turn}")

        for turn in min_turns:
            if bush_flows[turn] < min_path_flow:
                print(f"error with turn {turn}")
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
def _shift_flow(
    topological_order,
    L,
    U,
    min_path_successors,
    max_path_successors,
    derivatives,
    costs,
    label,
    turn_flows,
    bush_flows,
    capacities,
    ff_tts,
    global_in_turns,
    global_out_turns,
    bush_out_turns,
    tolls,
    pas_epsilon,
):
    lowest_order_link = 1
    # print('new run in shift flow')
    for j in topological_order[::-1]:
        if U[j] - L[j] > pas_epsilon:  # the shifts here need to be tighter so that
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
            ) = _get_branch_links(
                j,
                min_path_successors,
                max_path_successors,
                bush_flows,
                global_out_turns,
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
                _equalize_cost(
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
                    turn_flows,
                    bush_flows,
                    global_in_turns,
                    global_out_turns,
                    derivatives,
                    costs,
                    capacities,
                    ff_tts,
                    tolls,
                    pas_epsilon,
                )
                assert total_flow == min_path_flow + max_path_flow
                # print(f'updating tree between {end_link} and {j} with labels'
                #       f' {label[end_link], label[j]}')
                _update_trees(
                    label[end_link],
                    label[j] + 1,
                    L,
                    U,
                    min_path_successors,
                    max_path_successors,
                    topological_order,
                    costs,
                    bush_flows,
                    global_out_turns,
                    bush_out_turns,
                )
                # print(f'cost differences now {U[j] - L[j]}')
                assert abs(U[j] - L[j]) < 99999
            else:
                continue
            lowest_order_link = j

    return lowest_order_link
