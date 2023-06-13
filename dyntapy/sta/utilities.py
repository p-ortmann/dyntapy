#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
from numba import njit, prange
from numba.typed import List

from dyntapy.demand import InternalStaticDemand
from dyntapy.supply import Links
from dyntapy.graph_utils import (
    _make_in_links,
    _make_out_links,
    dijkstra_all,
    pred_to_paths,
)
from dyntapy.settings import parameters
from dyntapy.supply import Network

bpr_b = parameters.static_assignment.bpr_beta
bpr_a = parameters.static_assignment.bpr_alpha


@njit
def _bpr_cost_tolls(flows, capacities, ff_tts, tolls):
    number_of_links = len(flows)
    costs = np.empty(number_of_links, dtype=np.float64)
    for it in prange(number_of_links):
        toll = tolls[it]
        f = flows[it]
        c = capacities[it]
        ff_tt = ff_tts[it]
        assert c != 0
        costs[it] = _bpr_cost_single_toll(f, c, ff_tt, toll)
    return costs


@njit
def _bpr_cost(flows, capacities, ff_tts):
    number_of_links = len(flows)
    costs = np.empty(number_of_links, dtype=np.float64)
    for it in prange(number_of_links):
        f = flows[it]
        c = capacities[it]
        ff_tt = ff_tts[it]
        assert c != 0
        costs[it] = _bpr_cost_single(f, c, ff_tt)
    return costs


@njit
def _bpr_cost_single_toll(flow, capacity, ff_tt, toll):
    return toll + 1.0 * ff_tt + np.multiply(bpr_a, pow(flow / capacity, bpr_b)) * ff_tt


@njit
def _bpr_cost_single(flow, capacity, ff_tt):
    return 1.0 * ff_tt + np.multiply(bpr_a, pow(flow / capacity, bpr_b)) * ff_tt


@njit
def _bpr_derivative(flows, capacities, ff_tts):
    number_of_links = len(flows)
    derivatives = np.empty(number_of_links, dtype=np.float64)
    for it in prange(number_of_links):
        f = flows[it]
        c = capacities[it]
        ff_tt = ff_tts[it]
        assert c != 0
        derivatives[it] = ff_tt * bpr_a * bpr_b * (1 / c) * pow(f / c, bpr_b - 1)
    return derivatives


@njit
def _bpr_derivative_single(flow, capacity, ff_tt):
    return ff_tt * bpr_a * bpr_b * (1 / capacity) * pow(flow / capacity, bpr_b - 1)


@njit(parallel=True, nogil=True)
def aon(demand: InternalStaticDemand, costs, network: Network):
    out_links = network.nodes.out_links
    flows = np.zeros(len(costs))
    number_of_od_pairs = 0
    for i in demand.to_destinations.get_nnz_rows():
        number_of_od_pairs += demand.to_destinations.get_nnz(i).size
    ssp_costs = np.zeros(number_of_od_pairs)
    counter = 0
    for i in demand.to_destinations.get_nnz_rows():
        destinations = demand.to_destinations.get_nnz(i)
        demands = demand.to_destinations.get_row(i)
        distances, pred = dijkstra_all(
            costs, out_links, source=i, is_centroid=network.nodes.is_centroid
        )
        path_costs = np.empty(destinations.size, dtype=np.float32)
        for idx, dest in enumerate(destinations):
            path_costs[idx] = distances[dest]
            # TODO: Check for correctness
        paths = pred_to_paths(pred, i, destinations, out_links)
        for path, path_flow, path_cost in zip(paths, demands, path_costs):
            ssp_costs[counter] = path_cost
            counter += 1
            for link_id in path:
                flows[link_id] += path_flow
    return ssp_costs, flows


@njit
def generate_bushes(
    link_ff_times, from_nodes, to_nodes, out_links, demand, tot_links, is_centroid
):
    tot_nodes = out_links.get_nnz_rows().size
    tot_origins = demand.to_destinations.get_nnz_rows().size
    topological_orders = np.empty((tot_origins, tot_nodes), dtype=np.int64)
    distances = np.empty((tot_origins, tot_nodes), np.float64)
    links_in_bush = np.full((tot_origins, tot_links), False)
    assert demand.to_destinations.get_nnz_rows().size > 0
    for origin_id, origin in enumerate(demand.to_destinations.get_nnz_rows()):
        distances[origin_id], pred = dijkstra_all(
            costs=link_ff_times,
            out_links=out_links,
            source=origin,
            is_centroid=is_centroid,
        )
        topological_orders[origin_id] = np.argsort(distances[origin_id])
        label = np.argsort(topological_orders[origin_id])
        for link_id, (i, j) in enumerate(zip(from_nodes, to_nodes)):
            if label[j] > label[i]:
                links_in_bush[origin_id][link_id] = True
    return topological_orders, links_in_bush, distances


@njit
def generate_bushes_line_graph(
    turn_cost,
    from_link,
    to_link,
    in_turns,
    destination_links,
    tot_links,
):
    tot_destinations = destination_links.size
    topological_orders = np.empty((tot_destinations, tot_links), dtype=np.int64)
    distances = np.empty((tot_destinations, tot_links), np.float64)
    tot_turns = from_link.size
    turns_in_bush = np.full((tot_destinations, tot_turns), False)
    is_centroid = np.full(tot_links, False)
    is_centroid[destination_links] = True
    all_bush_in_turns = List()
    all_bush_out_turns = List()
    for destination_id, destination_link in enumerate(destination_links):
        distances[destination_id], pred = dijkstra_all(
            costs=turn_cost,
            out_links=in_turns,
            source=destination_link,
            is_centroid=is_centroid,
        )
        topological_orders[destination_id] = np.argsort(distances[destination_id])
        label = np.argsort(topological_orders[destination_id])
        for turn, (i, j) in enumerate(zip(from_link, to_link)):
            if label[j] < label[i]:
                turns_in_bush[destination_id][turn] = True

        bush_out_turns = _make_out_links(
            turns_in_bush[destination_id], from_link, to_link, tot_links
        )
        bush_in_turns = _make_in_links(
            turns_in_bush[destination_id], from_link, to_link, tot_links
        )
        all_bush_out_turns.append(bush_out_turns)
        all_bush_in_turns.append(bush_in_turns)

    return (
        topological_orders,
        turns_in_bush,
        distances,
        all_bush_in_turns,
        all_bush_out_turns,
    )


@njit
def _link_to_turn_cost_static(
    tot_turns, to_links, link_cost, turn_restriction, restricted_turn_cost=3600 / 3600
):
    turn_costs = np.zeros(tot_turns, dtype=np.float64)
    for turn in range(tot_turns):
        to_link = to_links[turn]
        if not turn_restriction[turn]:
            turn_costs[turn] = link_cost[to_link]
        else:
            turn_costs[turn] = max(restricted_turn_cost, link_cost[to_link])  #
            # large penalty for u turns
    return turn_costs


@njit
def _get_u_turn_turn_restrictions(tot_turns, from_node, to_node):
    turn_restrictions = np.full(tot_turns, False)
    for turn in range(tot_turns):
        if from_node[turn] == to_node[turn]:
            turn_restrictions[turn] = True
    return turn_restrictions


@njit
def beckmann(links: Links, flows):
    tot_links = links.capacity.size
    cap = links.capacity
    ff_tts = links.length / links.free_speed
    integrals = np.empty(tot_links, dtype=np.float64)
    for it in prange(tot_links):
        f = flows[it]
        c = cap[it]
        ff_tt = ff_tts[it]
        assert c != 0
        integrals[it] = ff_tt * ((bpr_a * c * pow(f / c, bpr_b + 1)) / (bpr_b + 1) + f)
    return np.sum(integrals)
