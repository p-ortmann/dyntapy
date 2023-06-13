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
from numba import njit, objmode, prange
from numba.typed import List

from dyntapy._context import iteration_states
from dyntapy.csr import BCSRMatrix, UI32CSRMatrix, csr_prep
from dyntapy.demand import InternalStaticDemand
from dyntapy.graph_utils import dijkstra_all, pred_to_paths
from dyntapy.results import StaticResult
from dyntapy.settings import debugging, debugging_full, parameters
from dyntapy.sta.equilibrate_bush import _equilibrate_bush, _update_bush
from dyntapy.sta.utilities import (
    beckmann,
    _bpr_cost_single_toll,
    _bpr_cost_tolls,
    _bpr_derivative,
    _link_to_turn_cost_static,
)
from dyntapy.supply import Network

epsilon = parameters.static_assignment.dial_b_cost_differences
dial_b_max_iterations = parameters.static_assignment.dial_b_max_iterations
commodity_type = "destination"  # returned multi commodity flow is origin based
gap_definition = (
    f"{epsilon=} converged destination - bushes divided by total origins, "
    "becomes 1 in equilibrium"
)

# as long as the functions take the same inputs and return the costs and derivatives
# respectively, any continuous function can be used.
cost_function = _bpr_cost_tolls
derivative_function = _bpr_derivative


@njit
def make_boolean_turn_csr(from_link, to_link, tot_links, is_allowed_turn):
    # bool for each turn
    # sparse structure for each out_turn

    tot_turns = np.uint32(len(to_link))
    fw_index_array = np.column_stack((from_link, np.arange(tot_turns, dtype=np.uint32)))
    bw_index_array = np.column_stack((to_link, np.arange(tot_turns, dtype=np.uint32)))

    val = np.copy(is_allowed_turn)
    val, col, row = csr_prep(
        fw_index_array, val, (tot_links, tot_turns), unsorted=False
    )
    is_out_turn = BCSRMatrix(val, col, row)
    val = np.copy(is_allowed_turn)
    val, col, row = csr_prep(bw_index_array, val, (tot_links, tot_turns))
    is_in_turn = BCSRMatrix(val, col, row)
    return is_out_turn, is_in_turn


@njit
def dial_b(
    network: Network, demand: InternalStaticDemand, store_iterations, tolls, eps=epsilon
):
    gaps = []
    to_links = network.turns.to_link
    link_ff_times = network.links.length / network.links.free_speed
    capacities = network.links.capacity
    (
        flows,
        bush_flows,
        topological_orders,
        turns_in_bush,
        bush_out_turns,
        last_indices,
    ) = _initial_loading(network, demand, tolls)
    turn_flows = np.sum(bush_flows, axis=0)
    link_costs = cost_function(
        capacities=network.links.capacity,
        ff_tts=link_ff_times,
        flows=flows,
        tolls=tolls,
    )
    turn_costs = _link_to_turn_cost_static(
        network.tot_turns,
        to_links,
        link_costs,
        turn_restriction=np.full(network.tot_turns, False),
    )
    derivatives = derivative_function(
        flows=flows, capacities=network.links.capacity, ff_tts=link_ff_times
    )
    turn_derivatives = np.zeros(network.tot_turns)
    for turn in range(network.tot_turns):
        turn_derivatives[turn] = derivatives[network.turns.to_link[turn]]
    iteration = 0

    destination_links = np.empty_like(demand.destinations)
    for _id, dest in enumerate(demand.destinations):
        destination_links[_id] = network.nodes.in_links.get_nnz(dest)[0]
    if debugging:
        print("Dial equilibration started")
    convergence_counter = 0
    while iteration < dial_b_max_iterations:
        if debugging and iteration > 0:
            print(f" previous convergence counter : {convergence_counter}")
        convergence_counter = 0
        if store_iterations:
            arr_gap = np.zeros(len(gaps))
            for idx, val in enumerate(gaps):
                arr_gap[idx] = val
            with objmode():
                print("storing iterations")
                iteration_states.append(
                    StaticResult(
                        link_costs,
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
            dest = destination_links[d_id]
            if debugging:
                print(
                    f"IN ITERATION: {iteration} with dest= {dest} and" f" d_id = {d_id}"
                )
            converged = False
            iterations_eq = 0
            while not converged:
                (
                    turn_flows,
                    bush_flows[d_id],
                    topological_orders[d_id][: last_indices[d_id]],
                    _,
                    converged,
                    bush_out_turns[d_id],
                    _,
                ) = _equilibrate_bush(
                    turn_costs,
                    bush_flows[d_id],
                    turn_flows,
                    dest,
                    demand.to_origins.get_nnz(demand.destinations[d_id]),
                    topological_orders[d_id][: last_indices[d_id]],
                    turn_derivatives,
                    capacities,
                    link_ff_times,
                    bush_out_turns[d_id],
                    eps,
                    network.links.out_turns,
                    network.links.in_turns,
                    to_links,
                    tolls,
                )
                if converged and iterations_eq == 0:
                    convergence_counter += 1
                iterations_eq += 1
        # print(f'number of converged bushes {convergence_counter} out of {len(
        # demand_dict)}')
        gaps.append(convergence_counter / demand.destinations.size)
        if convergence_counter == demand.destinations.size:
            break

        iteration = iteration + 1
        if iteration == dial_b_max_iterations:
            print("max iterations reached")

    print(f"solution found, Dial B in iteration {iteration}")
    gap_arr = np.empty(len(gaps), dtype=np.float64)
    for _id, val in enumerate(gaps):
        gap_arr[_id] = val
    link_destination_flows = np.zeros(
        (demand.destinations.size, network.tot_links), np.float64
    )
    tot_centroids = network.nodes.is_centroid.sum()
    for d_id in prange(demand.destinations.size):
        for link_id in range(network.tot_links):
            for turn_id in network.links.in_turns.get_nnz(link_id):
                link_destination_flows[d_id][link_id] += bush_flows[d_id][turn_id]
            if link_id < tot_centroids:
                # origins have no incoming turns
                for turn_id in network.links.out_turns.get_nnz(link_id):
                    link_destination_flows[d_id][link_id] += bush_flows[d_id][turn_id]

    link_costs = _bpr_cost_tolls(
        np.sum(link_destination_flows, axis=0),
        network.links.capacity,
        link_ff_times,
        tolls,
    )
    return link_costs, link_destination_flows, gap_definition, gap_arr


@njit()
def _initial_loading(network: Network, demand: InternalStaticDemand, tolls):
    tot_links = network.tot_links
    tot_turns = network.tot_turns
    link_ff_times = network.links.length / network.links.free_speed
    flows = np.zeros(tot_links)
    from_link = network.turns.from_link
    to_link = network.turns.to_link
    in_turns = network.links.in_turns
    is_centroid = np.full(tot_links, False)
    destination_links = np.empty_like(demand.destinations)
    for _id, dest in enumerate(demand.destinations):
        destination_links[_id] = network.nodes.in_links.get_nnz(dest)[0]
    tot_destinations = demand.destinations.size
    last_indices = np.empty(tot_destinations, np.uint32)
    bush_flows = np.zeros((tot_destinations, tot_turns), dtype=np.float64)
    topological_orders = np.empty((tot_destinations, tot_links), dtype=np.uint32)
    turns_in_bush = np.full((tot_destinations, tot_turns), False)

    costs = cost_function(
        capacities=network.links.capacity,
        ff_tts=link_ff_times,
        flows=flows,
        tolls=tolls,
    )
    turn_costs = _link_to_turn_cost_static(
        network.tot_turns,
        network.turns.to_link,
        costs,
        turn_restriction=np.full(network.tot_turns, False),
    )
    bush_out_turns = List()

    for d_id in prange(destination_links.size):
        destination_link = destination_links[d_id]
        dest = demand.destinations[d_id]
        origin_links = demand.to_origins.get_nnz(dest)
        distances, pred = dijkstra_all(
            costs=turn_costs,
            out_links=in_turns,
            source=destination_link,
            is_centroid=is_centroid,
        )
        paths = pred_to_paths(
            pred, destination_link, origin_links, network.links.out_turns, reverse=True
        )
        topological_orders[d_id] = np.argsort(distances)
        sorted_dist = np.sort(distances)
        last_index = np.argwhere(sorted_dist == np.inf).flatten()[0]
        last_indices[d_id] = last_index
        label = np.argsort(topological_orders[d_id])
        for turn, (i, j) in enumerate(zip(from_link, to_link)):
            if label[j] < label[i] < last_index:  # don't include turns
                # that start from links that cannot reach the destination
                turns_in_bush[d_id][turn] = True
        for path, path_flow in zip(paths, demand.to_origins.get_row(dest)):
            for turn_id in path:
                flows[to_link[turn_id]] += path_flow
                bush_flows[d_id][turn_id] += path_flow
        bush_out_turn, _ = make_boolean_turn_csr(
            from_link, to_link, tot_links, turns_in_bush[d_id]
        )
        bush_out_turns.append(bush_out_turn)

    return (
        flows,
        bush_flows,
        topological_orders,
        turns_in_bush,
        bush_out_turns,
        last_indices,
    )
