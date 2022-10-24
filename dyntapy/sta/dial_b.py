#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
# There are no extensive comments in this module although they definitely would be
# helpful if looking at this for educational purposes, I'd recommend to consult
# Steven Boyles upcoming Book Transport Network Analysis and related course notes in
# github (https://sboyles.github.io/teaching/ce392c/10-bushbased.pdf)  for background
# and explanations and the more technical paper from Dial (Algorithm B: Accurate
# Traffic Equilibrium (and How to Bobtail Frank-Wolfe) as a reference and for
# understanding the more nitty-gritty details of implementation.
from heapq import heappop, heappush

import numpy as np
from numba import njit, prange, objmode
from dyntapy._context import iteration_states
from dyntapy.demand import InternalStaticDemand
from dyntapy.graph_utils import (
    dijkstra_all,
    _make_in_links,
    _make_out_links,
    pred_to_paths,
    _get_link_id,
)
from dyntapy.settings import parameters, debugging
from dyntapy.sta.equilibrate_bush import __equilibrate_bush
from dyntapy.sta.utilities import __bpr_cost_tolls, __bpr_derivative, \
    __link_to_turn_cost_static
from dyntapy.supply import Network
from dyntapy.results import StaticResult

epsilon = parameters.static_assignment.dial_b_cost_differences
dial_b_max_iterations = parameters.static_assignment.dial_b_max_iterations
commodity_type = "destination"  # returned multi commodity flow is origin based
gap_definition = (
    f"{epsilon=} converged destination - bushes divided by total origins, "
    "becomes 1 in equilibrium"
)

# as long as the functions take the same inputs and return the costs and derivatives
# respectively, any continuous function can be used.
cost_function = __bpr_cost_tolls
derivative_function = __bpr_derivative


@njit
def dial_b(network: Network, demand: InternalStaticDemand, store_iterations, tolls):
    gaps = []
    from_links = network.turns.from_link
    to_links = network.turns.to_link
    link_ff_times = network.links.length / network.links.free_speed
    turn_costs = __link_to_turn_cost_static(network.tot_turns, from_links,link_ff_times,
                                            turn_restriction=np.full(network.tot_turns,
                                                                     False))
    flows, bush_flows, topological_orders, turns_in_bush = __initial_loading(
        network, demand, tolls
    )
    costs = cost_function(
        capacities=network.links.capacity, ff_tts=link_ff_times, flows=flows,
        tolls=tolls
    )
    derivatives = derivative_function(
        flows=flows, capacities=network.links.capacity, ff_tts=link_ff_times
    )
    iteration = 0
    if debugging:
        print("Dial equilibration started")
    while iteration < dial_b_max_iterations:
        convergence_counter = 0
        if debugging:
            print(f" convergence counter : {convergence_counter}")
        if store_iterations:
            arr_gap = np.zeros(len(gaps))
            for idx, val in enumerate(gaps):
                arr_gap[idx] = val
            with objmode():
                print("storing iterations")
                iteration_states.append(
                    StaticResult(
                        costs,
                        flows,
                        demand.origins,
                        demand.destinations,
                        gap_definition=gap_definition,
                        gap=arr_gap,
                        origin_flows=bush_flows,
                    )
                )
        for d_id in range(demand.destinations.size):
            # creating a bush for each origin
            # we're not using the sparse matrices here
            # they cannot be changed once created
            d_id = np.uint32(d_id)
            dest = demand.to_destinations.get_nnz_rows()[d_id]
            if debugging:
                print(
                    f"IN ITERATION: {convergence_counter} with dest= {dest} and"
                    f" d_id = {d_id}"
                )
            bush_out_links = _make_out_links(
                turns_in_bush[d_id],
                from_links,
                to_links,
                tot_nodes=network.tot_links,
            )
            bush_in_links = _make_in_links(
                turns_in_bush[d_id],
                from_links,
                to_links,
                tot_nodes=network.tot_turns,
            )
            token = 0
            while True:
                if debugging:
                    print(f"equilibrating dest {dest}")
                (
                    flows,
                    bush_flows[d_id],
                    turns_in_bush[d_id],
                    converged_without_shifts,
                    L,
                    U,
                    bush_out_turns,
                    bush_in_turns,
                ) = __equilibrate_bush(
                    costs,
                    bush_flows=bush_flows[d_id],
                    origin=dest,
                    flows=flows,
                    topological_order=topological_orders[d_id],
                    derivatives=derivatives,
                    links_in_bush=turns_in_bush[d_id],
                    capacities=network.links.capacity,
                    ff_tts=link_ff_times,
                    bush_out_links=bush_out_links,
                    bush_in_links=bush_in_links,
                    epsilon=epsilon,
                    global_out_links=network.nodes.out_links,
                    tot_links=network.tot_links,
                    to_node=network.links.to_node,
                    from_node=network.links.from_node,
                )
                if converged_without_shifts:
                    if token >= 1:
                        if token == 1:
                            convergence_counter += 1

                        if debugging:
                            print(
                                f" no shifts after edges added for dest {dest},"
                                f"moving on"
                            )
                        break

                for k in topological_orders[d_id]:
                    assert k in L
                edges_added, topological_orders[d_id] = __update_bush(
                    L=L,
                    bush_edges=turns_in_bush[d_id],
                    costs=costs,
                    from_node=network.links.from_node,
                    to_node=network.links.to_node,
                    bush_out_links=bush_out_links,
                    bush_in_links=bush_in_links,
                    bush_flows=bush_flows[d_id],
                    tot_nodes=network.tot_nodes,
                    origin=dest,
                    global_out_links_csr=network.nodes.out_links,
                )
                assert len(set(topological_orders[d_id])) == network.tot_nodes
                if not edges_added:
                    if converged_without_shifts and token == 0:
                        convergence_counter += 1
                        if debugging:
                            print(
                                f" dest {dest} is converged and no edges were added, "
                                f"moving on"
                            )
                    break
                else:
                    if debugging:
                        print("edges added")
                    pass

                token += 1
        # print(f'number of converged bushes {convergence_counter} out of {len(
        # demand_dict)}')
        gaps.append(convergence_counter / demand.to_destinations.get_nnz_rows().size)
        if convergence_counter == demand.to_destinations.get_nnz_rows().size:
            break
        iteration = iteration + 1
        if iteration == dial_b_max_iterations:
            print("max iterations reached")

    gap_arr = np.empty(len(gaps), dtype=np.float64)
    for _id, val in enumerate(gaps):
        gap_arr[_id] = val
    return costs, bush_flows, gap_definition, gap_arr


@njit
def __update_bush(
        L,
        bush_edges,
        costs,
        from_node,
        to_node,
        bush_out_links,
        bush_in_links,
        bush_flows,
        tot_nodes,
        destination,
        global_out_links_csr,
):
    new_turns_added = False
    for link_id, link_tuple in enumerate(zip(from_node, to_node)):
        i, j = link_tuple[0], link_tuple[1]
        if L[i] + costs[link_id] < L[j] - epsilon:
            # current edges is a shortcut edge
            # pos2 = np.argwhere(topological_order == j)[0][0]
            # pos1 = np.argwhere(topological_order == i)[0][0]
            # assert pos2>pos1
            # if pos1 > pos2:
            #    print('something wrong with labels or shifting')
            opposite_exists = False
            try:
                opposite_link = _get_link_id(j, i, global_out_links_csr)
                opposite_exists = True
            except Exception:
                pass

            if opposite_exists:
                flow_opposite_edge = bush_flows[opposite_link]
                if flow_opposite_edge > 0:
                    # cannot remove this edge, it's loaded! adding the opposite would
                    # yield a graph that cannot be topologically sorted this may
                    # happen with very short links and comparatively large epsilon
                    # values
                    continue
            bush_edges[link_id] = True
            # adding an in_link i to j
            in_links_j = np.empty(
                (bush_in_links[j].shape[0] + 1, bush_in_links[j].shape[1]),
                dtype=np.int64,
            )
            in_links_j[:-1] = bush_in_links[j]
            in_links_j[-1] = np.int64(i), link_id
            bush_in_links[j] = in_links_j

            # adding an out_link j to i
            out_links_i = np.empty(
                (bush_out_links[i].shape[0] + 1, bush_out_links[i].shape[1]),
                dtype=np.int64,
            )
            out_links_i[:-1] = bush_out_links[i]
            out_links_i[-1] = np.int64(j), link_id
            bush_out_links[i] = out_links_i
            new_edges_added = True

            if opposite_exists:
                if bush_edges[opposite_link]:
                    bush_edges[opposite_link] = False
                    # removing an in_link j from i
                    in_links_i = np.empty(
                        (bush_in_links[i].shape[0] - 1, bush_in_links[i].shape[1]),
                        dtype=np.int64,
                    )
                    idx = 0
                    for _j, _link_id in bush_in_links[i]:
                        if _j != j:
                            in_links_i[idx] = _j, _link_id
                            idx += 1
                    bush_in_links[i] = in_links_i

                    # removing an out_link i from j
                    out_links_j = np.empty(
                        (bush_out_links[i].shape[0] - 1, bush_out_links[i].shape[1]),
                        dtype=np.int64,
                    )
                    idx = 0
                    for _i, _link_id in bush_out_links[j]:
                        if _i != i:
                            out_links_j[idx] = _i, _link_id
                            idx += 1
                    bush_out_links[j] = out_links_j
    return new_edges_added, topological_sort(
        bush_out_links, bush_in_links, tot_nodes, origin
    )


@njit
def topological_sort(forward_star, backward_star, tot_nodes, origin):
    # topological sort on a graph assuming that backward and forward stars form
    # a connected DAG, (Directed Acyclic Graph)
    order = np.zeros(tot_nodes, dtype=np.uint32)
    order[0] = origin
    my_heap = []
    my_heap.append((np.uint32(0), np.uint32(origin)))
    visited_node = np.full(tot_nodes, False)
    idx = 0
    while my_heap:
        my_tuple = heappop(my_heap)
        pos = my_tuple[0]
        i = my_tuple[1]
        visited_node[i] = True
        order[idx] = i
        idx = idx + 1
        pos_count = pos
        for arr in forward_star[i]:
            j = arr[0]
            visited_all_nodes_in_backward_star = True
            for arr2 in backward_star[j]:
                i2 = arr2[0]
                if not visited_node[i2]:
                    visited_all_nodes_in_backward_star = False
            if visited_all_nodes_in_backward_star:
                pos_count += 1
                heappush(my_heap, (np.uint32(pos_count), np.uint32(j)))
    assert len(set(order)) == tot_nodes  # if this fails forward and backward stars do
    # not form DAG
    return order


@njit()
def __initial_loading(network: Network, demand: InternalStaticDemand, tolls):
    tot_links = network.tot_links
    tot_turns = network.tot_turns
    link_ff_times = network.links.length / network.links.free_speed
    flows = np.zeros(tot_links)
    from_link = network.turns.from_link
    to_link = network.turns.to_link
    in_turns = network.links.in_turns
    is_centroid = np.full(tot_links, False)

    origin_links = np.empty_like(demand.origins)
    for _id, origin in enumerate(demand.origins):
        origin_links[_id] = network.nodes.out_links.get_nnz(origin)[0]
    destination_links = np.empty_like(demand.destinations)
    for _id, dest in enumerate(demand.destinations):
        destination_links[_id] = network.nodes.in_links.get_nnz(dest)[0]
    tot_destinations = demand.destinations.size
    bush_flows = np.zeros((tot_destinations, tot_turns), dtype=np.float64)
    topological_orders = np.empty((tot_destinations, tot_links), dtype=np.uint32)
    turns_in_bush = np.full(
        (destination_links.size, tot_turns), False

    )


    costs = cost_function(
        capacities=network.links.capacity, ff_tts=link_ff_times, flows=flows,
        tolls = tolls
    )
    turn_costs = __link_to_turn_cost_static(network.tot_turns, network.turns.from_link,
                                            costs, turn_restriction=np.full(
        network.tot_turns,
                                                                     False))

    for d_id in prange(destination_links.size):
        destination_link = destination_links[d_id]
        distances, pred = dijkstra_all(
            costs=turn_costs, out_links=in_turns, source=destination_link,
            is_centroid=is_centroid
        )
        paths = pred_to_paths(pred, destination_link, origin_links,
                              network.links.out_turns)
        topological_orders[d_id] = np.argsort(distances)
        label = np.argsort(topological_orders[d_id])
        for turn, (i, j) in enumerate(zip(from_link, to_link)):
            if label[j] < label[i]:
                turns_in_bush[d_id][turn] = True
        for path, path_flow in zip(paths, demand.to_origins.get_row(d_id)):
            for turn_id in path:
                flows[from_link[turn_id]] += path_flow
                bush_flows[d_id][turn_id] += path_flow

    return flows, bush_flows, topological_orders, turns_in_bush


