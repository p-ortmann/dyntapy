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
    make_in_links,
    make_out_links,
    pred_to_paths,
)
from dyntapy.settings import parameters
from dyntapy.sta.equilibrate_bush import __equilibrate_bush
from dyntapy.sta.utilities import __bpr_cost, __bpr_derivative
from dyntapy.supply import Network
from dyntapy.results import StaticResult

epsilon = parameters.static_assignment.dial_b_cost_differences
dial_b_max_iterations = parameters.static_assignment.dial_b_max_iterations

commodity_type = "origin"  # returned multi commodity flow is origin based
gap_definition = (
    f"{epsilon=} converged origin - bushes divided by total origins, "
    "becomes 1 in equilibrium"
)


@njit
def dial_b(network: Network, demand: InternalStaticDemand, store_iterations):
    gaps = []
    from_nodes = network.links.from_node
    to_nodes = network.links.to_node
    link_ff_times = network.links.length / network.links.free_speed
    flows, bush_flows, topological_orders, links_in_bush = __initial_loading(
        network, demand
    )
    costs = __bpr_cost(
        capacities=network.links.capacity, ff_tts=link_ff_times, flows=flows
    )
    derivatives = __bpr_derivative(
        flows=flows, capacities=network.links.capacity, ff_tts=link_ff_times
    )
    iteration = 0
    print("Dial equilibration started")
    while iteration < dial_b_max_iterations:
        convergence_counter = 0
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
        for bush_id in range(demand.to_destinations.get_nnz_rows().size):
            # creating a bush for each origin
            # we're not using the sparse matrices here
            # they cannot be changed once created
            bush_id = np.uint32(bush_id)
            bush = demand.to_destinations.get_nnz_rows()[bush_id]
            bush_out_links = make_out_links(
                links_in_bush[bush_id],
                from_nodes,
                to_nodes,
                tot_nodes=network.tot_nodes,
            )
            bush_in_links = make_in_links(
                links_in_bush[bush_id],
                from_nodes,
                to_nodes,
                tot_nodes=network.tot_nodes,
            )
            token = 0
            while True:
                # print(f'equilibrating bush {bush}')
                (
                    flows,
                    bush_flows[bush_id],
                    links_in_bush[bush_id],
                    converged_without_shifts,
                    L,
                    U,
                    bush_out_links,
                    bush_in_links,
                ) = __equilibrate_bush(
                    costs,
                    bush_flows=bush_flows[bush_id],
                    origin=bush,
                    flows=flows,
                    topological_order=topological_orders[bush_id],
                    derivatives=derivatives,
                    links_in_bush=links_in_bush[bush_id],
                    capacities=network.links.capacity,
                    ff_tts=link_ff_times,
                    bust_out_links=bush_out_links,
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
                        # print(f' no shifts after edges added for bush {bush},
                        # moving on')
                        break

                for k in topological_orders[bush_id]:
                    assert k in L
                edges_added, topological_orders[bush_id] = __update_bush(
                    L=L,
                    bush_edges=links_in_bush[bush_id],
                    costs=costs,
                    from_node=network.links.from_node,
                    to_node=network.links.to_node,
                    bush_out_links=bush_out_links,
                    bush_in_links=bush_in_links,
                    bush_flows=bush_flows[bush_id],
                    tot_nodes=network.tot_nodes,
                    origin=bush,
                )
                if not edges_added:
                    if converged_without_shifts and token == 0:
                        convergence_counter += 1
                    # print(f' bush {bush} is converged and no edges were added,
                    # moving on')
                    break
                else:
                    pass
                    # print('edges added')
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
    origin,
):
    new_edges_added = False
    for link_id, link_tuple in enumerate(zip(from_node, to_node)):
        i, j = link_tuple[0], link_tuple[1]
        if L[i] + costs[link_id] < L[j] - epsilon:
            # pos2 = np.argwhere(topological_order == j)[0][0]
            # pos1 = np.argwhere(topological_order == i)[0][0]
            # assert pos2>pos1
            # if pos1 > pos2:
            #    print('something wrong with labels or shifting')
            flow_opposite_edge = bush_flows[link_id]
            if flow_opposite_edge > epsilon:
                # cannot remove this edge, it's loaded! adding the opposite would
                # yield a graph that cannot be topologically sorted this may
                # happen with very short links and comparatively large epsilon
                # values
                continue
            bush_edges[link_id] = True
            in_links_j = np.empty(
                (bush_in_links[j].shape[0] + 1, bush_in_links[j].shape[1]),
                dtype=np.int64,
            )
            in_links_j[:-1] = bush_in_links[j]
            in_links_j[-1] = np.int64(i), link_id
            bush_in_links[j] = in_links_j
            out_links_j = np.empty(
                (bush_out_links[i].shape[0] + 1, bush_out_links[i].shape[1]),
                dtype=np.int64,
            )
            out_links_j[:-1] = bush_out_links[i]
            out_links_j[-1] = np.int64(j), link_id
            bush_out_links[i] = out_links_j
            new_edges_added = True

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
    return order


@njit()
def __initial_loading(network: Network, demand: InternalStaticDemand):
    tot_links = network.tot_links
    link_capacities = network.links.capacity
    link_ff_times = network.links.length / network.links.free_speed
    flows = np.zeros(tot_links)
    to_node = network.links.to_node
    from_node = network.links.from_node
    out_links = network.nodes.out_links
    is_centroid = network.nodes.is_centroid
    tot_origins = demand.to_destinations.get_nnz_rows().size
    tot_nodes = network.tot_nodes
    bush_flows = np.zeros((tot_origins, tot_links), dtype=np.float32)
    topological_orders = np.empty((tot_origins, tot_nodes), dtype=np.uint32)
    links_in_bush = np.full(
        (demand.to_destinations.get_nnz_rows().size, tot_links), False
    )
    for origin_id in prange(demand.to_destinations.get_nnz_rows().size):
        origin = demand.to_destinations.get_nnz_rows()[origin_id]
        bush_flows[origin_id] = np.zeros(tot_links)
        costs = __bpr_cost(
            capacities=link_capacities, ff_tts=link_ff_times, flows=flows
        )
        destinations = demand.to_destinations.get_nnz(origin)
        demands = demand.to_destinations.get_row(origin)
        distances, pred = dijkstra_all(
            costs=costs, out_links=out_links, source=origin, is_centroid=is_centroid
        )
        paths = pred_to_paths(pred, origin, destinations, out_links)
        topological_orders[origin_id] = np.argsort(distances)
        label = np.empty_like(topological_orders[0])
        for _id, j in enumerate(topological_orders[origin_id]):
            label[j] = _id
        for link_id, (i, j) in enumerate(zip(from_node, to_node)):
            if label[j] > label[i]:
                links_in_bush[origin_id][link_id] = True
        for path, path_flow in zip(paths, demands):
            for link_id in path:
                flows[link_id] += path_flow
                bush_flows[origin_id][link_id] += path_flow

    return flows, bush_flows, topological_orders, links_in_bush
