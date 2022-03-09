#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from dataclasses import dataclass, field
import numpy as np
from numba import njit, prange
from heapq import heappop, heappush
from numba.typed.typedlist import List as NumbaList

from dyntapy.demand import InternalStaticDemand
from dyntapy.supply import Network
from dyntapy.graph_utils import dijkstra_all
from dyntapy.csr import UI32CSRMatrix
from dyntapy.dta.i_ltm_cls import ILTMState


@dataclass
class StaticResult:
    link_costs: np.ndarray
    flows: np.ndarray
    origins: np.ndarray
    destinations: np.ndarray
    origin_flows: np.ndarray = None
    destination_flows: np.ndarray = None
    skim: np.ndarray = None
    gap_definition: str = None
    gap: np.ndarray = None
    od_flows: list = None


@dataclass
class DynamicResult:
    link_costs: np.ndarray
    cvn_up: np.ndarray
    cvn_down: np.ndarray
    con_up: np.ndarray
    con_down: np.ndarray
    turning_fractions: np.ndarray
    turn_costs: np.ndarray
    flows: np.ndarray
    commodity_type: str
    origins: np.ndarray
    destinations: np.ndarray
    skim: np.ndarray = None
    gap_definition: str = None
    iterations: np.ndarray = None


@njit(parallel=True, cache=True)
def get_skim(link_costs, demand: InternalStaticDemand, network: Network):
    # get skim matrices in dense format such that skim[0, 10] is the impedance between
    # the first and the eleventh zone corresponding to nodes demand.origins[0] and
    # demand.destinations[10] in the graph
    is_centroid = network.nodes.is_centroid
    out_links = network.nodes.out_links
    skim = np.empty((demand.origins.size, demand.destinations.size))

    for origin_id in prange(demand.origins.size):
        origin = demand.origins[origin_id]
        dist, _ = dijkstra_all(link_costs, out_links, origin, is_centroid)
        for destination_id, destination in enumerate(demand.destinations):
            skim[origin_id, destination_id] = dist[destination]
    return skim


def get_od_flows(assignment, result: StaticResult, return_as_matrix=False):
    # add interval for dynamic
    # add handling of dynamic assignment result
    if result.origin_flows is not None:
        commodity_type = "origin"
        od_flows = _get_od_flows(
            result.origin_flows,
            commodity_type,
            assignment.internal_demand,
            assignment.internal_network,
        )
    elif result.destination_flows is not None:
        commodity_type = "destination"
        od_flows = _get_od_flows(
            result.destination_flows,
            commodity_type,
            assignment.internal_demand,
            assignment.internal_network,
        )
    else:
        raise ValueError(
            "neither origin or destination flows specified in result " "object"
        )
    # copying from numba list to python list
    if return_as_matrix:
        # may cause out of memory errors for large assignments
        od_flow_mat = np.zeros(
            (
                assignment.internal_demand.origins.size,
                assignment.internal_demand.destinations.size,
                assignment.internal_network.tot_links,
            ),
            dtype=np.float32,
        )

        @njit
        def fill_matrix():
            for link in range(assignment.internal_network.tot_links):
                for origin, destination, flow in od_flows[link]:
                    od_flow_mat[int(origin), int(destination), link] = np.float32(flow)

        fill_matrix()
        return od_flow_mat
    else:
        od_flows = [
            [(int(tup[0]), int(tup[1]), np.round(tup[2], decimals=4)) for tup in link]
            for link in od_flows
        ]
        return od_flows


def get_selected_link_analysis(assignment, od_flows, link):
    # has not been optimised for performance
    relevant_ods = [
        (origin, destination) for origin, destination, flow in od_flows[link]
    ]
    link_od_flows = [[] for _ in range(assignment.internal_network.tot_links)]
    link_od_flows[link] = od_flows[link]
    # for each od we need to determine the in and out flows surrounding a link
    # and then take the proportional cut from the link in question
    out_links = assignment.internal_network.nodes.out_links
    in_links = assignment.internal_network.nodes.in_links
    from_node = assignment.internal_network.links.from_node[link]
    to_node = assignment.internal_network.links.to_node[link]
    _calc_sla(
        relevant_ods,
        in_links,
        out_links,
        to_node,
        link_od_flows,
        od_flows,
        direction="downstream",
    )
    _calc_sla(
        relevant_ods,
        in_links,
        out_links,
        from_node,
        link_od_flows,
        od_flows,
        direction="upstream",
    )
    flows = np.zeros(assignment.internal_network.tot_links, dtype=np.float64)
    for link in range(len(link_od_flows)):
        for _, _, flow in link_od_flows[link]:
            flows[link] += flow
    return link_od_flows


def _calc_sla(
    relevant_ods,
    in_links,
    out_links,
    start_node,
    link_od_flows,
    od_flows,
    direction="downstream",
):
    threshold = 0.0001
    tot_od_in_flow = np.zeros(len(relevant_ods), np.float32)
    tot_od_out_flow = np.zeros(len(relevant_ods), np.float32)
    nodes_to_process = [(0, start_node)]

    def get_total_node_flow(adjacency_csr, commodity_flows, result_arr, cur_node):
        for _id, (loaded_origin, loaded_destination) in enumerate(relevant_ods):
            for link in adjacency_csr.get_nnz(cur_node):
                for origin, destination, flow in commodity_flows[link]:
                    if origin == loaded_origin and destination == loaded_destination:
                        result_arr[_id] += flow

    def settle_flows(adjacency_csr, settled_flows, tot_od_flows, tot_settled_flows):
        for _node, link in zip(
            adjacency_csr.get_row(node), adjacency_csr.get_nnz(node)
        ):
            # if this gets called a second time on the same node results should
            # overwritten
            settled_flows[link] = []
            tot_settled_flow_via_link = 0
            for _id, (loaded_origin, loaded_destination) in enumerate(relevant_ods):
                for origin, destination, flow in od_flows[link]:
                    if loaded_origin == origin and loaded_destination == destination:
                        settled_flow_via_link = (
                            flow / tot_od_flows[_id] * tot_settled_flows[_id]
                        )
                        if settled_flow_via_link > threshold:
                            settled_flows[link].append(
                                (origin, destination, settled_flow_via_link)
                            )
                            tot_settled_flow_via_link += settled_flow_via_link
            if tot_settled_flow_via_link > threshold:
                heappush(nodes_to_process, (-tot_settled_flow_via_link, _node))

    if direction == "upstream":
        while nodes_to_process:
            _, node = heappop(nodes_to_process)
            tot_od_out_flow[:] = 0
            tot_od_in_flow[:] = 0
            get_total_node_flow(out_links, link_od_flows, tot_od_out_flow, node)
            get_total_node_flow(in_links, od_flows, tot_od_in_flow, node)
            settle_flows(in_links, link_od_flows, tot_od_in_flow, tot_od_out_flow)
    elif direction == "downstream":
        while nodes_to_process:
            _, node = heappop(nodes_to_process)
            tot_od_out_flow[:] = 0
            tot_od_in_flow[:] = 0
            get_total_node_flow(out_links, od_flows, tot_od_out_flow, node)
            get_total_node_flow(in_links, link_od_flows, tot_od_in_flow, node)
            settle_flows(out_links, link_od_flows, tot_od_out_flow, tot_od_in_flow)
    else:
        raise ValueError


@njit
def _get_od_flows(
    multi_commodity_flows: np.ndarray,
    commodity_type: str,
    demand: InternalStaticDemand,
    network: Network,
    threshold=0.0001,
):
    # output should be a list of lists: sla[link] = [(origin, destination, value),
    # (..,..,..), ....]
    sla = NumbaList()
    for link in range(network.tot_links):
        list_hull = NumbaList()
        list_hull.append((0.0, 0.0, 0.0))
        sla.append(list_hull)
        list_hull.pop(0)
        # empty list typing in numba ..
    origins = demand.origins
    tot_origins = demand.origins.size
    destinations = demand.destinations
    in_links: UI32CSRMatrix = network.nodes.in_links
    out_links: UI32CSRMatrix = network.nodes.out_links
    # out_links: UI32CSRMatrix = network.nodes.out_links
    assert commodity_type in ["origin", "destination"]
    if commodity_type == "origin":
        for destination in destinations:
            nodes_to_process = []  # using reflected list here since it doesn't work
            # for typed yet
            nodes_to_process.append((0.0, 0.0))  # here for type inference
            nodes_to_process.pop(0)
            origin_flows_for_destination = np.zeros(
                (origins.size, network.tot_links), dtype=np.float64
            )  # origin flows on each edge for the current destination
            for in_link, node in zip(
                in_links.get_nnz(destination), in_links.get_row(destination)
            ):
                # sink connectors to the destination must be routing towards it
                assert network.links.link_type[in_link] == -1
                origin_flows_for_destination[:, in_link] = multi_commodity_flows[
                    :, in_link
                ]
                flow = np.sum(multi_commodity_flows[:, in_link])
                if flow > threshold:
                    heap_item = (
                        np.float64(-flow),
                        np.float64(node),
                    )
                    heappush(nodes_to_process, heap_item)
            while len(nodes_to_process) > 0:
                _propagate_flows(
                    origin_flows_for_destination,
                    in_links,
                    out_links,
                    multi_commodity_flows,
                    tot_origins,
                    nodes_to_process,
                    threshold,
                    network.nodes.is_centroid,
                )
            for origin_id, origin in enumerate(origins):
                for link in range(network.tot_links):
                    flow = origin_flows_for_destination[origin_id, link]
                    if flow > threshold:
                        sla[link].append(
                            (
                                np.float64(origin),
                                np.float64(destination),
                                np.float64(flow),
                            )
                        )
    return sla


@njit()
def _propagate_flows(
    origin_flows_for_destination,
    in_links,
    out_links,
    origin_flows,
    tot_origins,
    nodes_to_process,
    threshold,
    is_centroid,
):
    heap_item = heappop(nodes_to_process)
    node = np.uint32(heap_item[1])
    if is_centroid[node]:
        return 0
    # origin_flows_for_destination are fixed to a particular OD
    # which destination that is, i:s not relevant here
    # commodity flows are the full origin flows not just for the destination in question
    # the function propagates to the upstream links based on proportionality
    tot_out_flow_for_destination = np.zeros(tot_origins)
    # first we get the total outgoing OD flows
    for out_link in out_links.get_nnz(node):
        tot_out_flow_for_destination += origin_flows_for_destination[:, out_link]
    tot_in_flows = np.zeros(tot_origins)
    # calculate total incoming origin flows
    for in_link in in_links.get_nnz(node):
        tot_in_flows += origin_flows[:, in_link]
        # distribute upstream proportionally
        # calculating difference to previous state and updating flows
    for in_link, in_node in zip(in_links.get_nnz(node), in_links.get_row(node)):
        delta = 0
        for origin_id in range(tot_origins):
            if tot_out_flow_for_destination[origin_id] > 0:
                flow = (
                    origin_flows[origin_id, in_link] / tot_in_flows[origin_id]
                ) * tot_out_flow_for_destination[origin_id]
                # invalid values here hint at running across a centroid which
                # shouldn't happen
                delta += flow - origin_flows_for_destination[origin_id, in_link]
                origin_flows_for_destination[origin_id, in_link] = (
                    origin_flows[origin_id, in_link]
                    / tot_in_flows[origin_id]
                    * tot_out_flow_for_destination[origin_id]
                )
        if delta > threshold and not is_centroid[node]:
            heappush(nodes_to_process, (np.float64(-delta), np.float64(in_node)))


@njit(cache=True, parallel=True)
def cvn_to_flows(cvn_down):
    """

    Parameters
    ----------
    cvn :

    Returns
    -------

    """
    tot_time_steps = cvn_down.shape[0]
    tot_links = cvn_down.shape[1]
    cvn_down = np.sum(cvn_down, axis=2)
    flows = np.zeros((tot_time_steps, tot_links), dtype=np.float32)
    flows[0, :] = cvn_down[0, :]
    for time in prange(1, tot_time_steps):
        flows[time, :] = np.abs(-cvn_down[time - 1, :] + cvn_down[time, :])
    return flows
