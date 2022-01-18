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
from numba import njit
from numba.typed import List

from dyntapy.demand import InternalStaticDemand
from dyntapy.graph_utils import dijkstra_all, pred_to_paths
from dyntapy.settings import parameters
from dyntapy.supply import Network

bpr_b = parameters.static_assignment.bpr_beta
bpr_a = parameters.static_assignment.bpr_alpha


@njit
def __bpr_cost(flows, capacities, ff_tts):
    number_of_links = len(flows)
    costs = np.empty(number_of_links, dtype=np.float32)
    for it, (f, c, ff_tt) in enumerate(zip(flows, capacities, ff_tts)):
        assert c != 0
        costs[it] = __bpr_cost_single(f, c, ff_tt)
    return costs


@njit
def __bpr_cost_single(flow, capacity, ff_tt):
    return 1.0 * ff_tt + np.multiply(bpr_a, pow(flow / capacity, bpr_b)) * ff_tt


@njit
def __bpr_derivative(flows, capacities, ff_tts):
    number_of_links = len(flows)
    derivatives = np.empty(number_of_links)
    for it, (f, c, ff_tt) in enumerate(zip(flows, capacities, ff_tts)):
        assert c != 0
        derivatives[it] = ff_tt * bpr_a * bpr_b * (1 / c) * pow(f / c, bpr_b - 1)
    return derivatives


@njit
def __bpr_derivative_single(flow, capacity, ff_tt):
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
def __valid_edges(from_node, to_node, label):
    """

    Parameters
    ----------
    edge_map : dictionary keyed by (i,j) with edge index as value
    distances : dictionary of distances keyed by node ids

    Returns
    -------

    """
    edges = List()
    for (i, j) in zip(from_node, to_node):
        if label[j] > label[i]:
            edges.append((i, j))
    return edges
