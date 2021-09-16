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
from dyntapy.sta.demand import Demand
from numba.typed import List
from dyntapy.sta.algorithms.graph_utils import __shortest_path, __pred_to_epath
from dyntapy.settings import static_parameters
from numba import njit

bpr_b = static_parameters.assignment.bpr_beta
bpr_a = static_parameters.assignment.bpr_alpha


def calculate_costs(link_capacities, link_ff_times, link_flows, method='bpr'):
    if method == 'bpr':
        costs = __bpr_cost(flows=link_flows, capacities=link_capacities, ff_tts=link_ff_times)
    return costs


@njit
def __bpr_cost(flows, capacities, ff_tts):
    number_of_links = len(flows)
    costs = np.empty(number_of_links)
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

    # derivatives = np.apply_along_axis(bpr_derivative, 0, link_flows, capacities=link_capacities,
    #                                  ff_tt=link_ff_times, bpr_alpha=bpr_a, bpr_beta=bpr_b)
    # return derivatives


@njit(parallel=True, nogil=True)
def aon(demand: Demand, costs, forward_star, edge_map, number_of_od_pairs, node_order):
    flows = np.zeros(len(costs))
    ssp_costs = np.zeros(number_of_od_pairs)
    counter = 0
    for i in demand.to_destinations.get_nnz_rows():
        destinations = demand.to_destinations.get_nnz(i)
        demands = demand.to_destinations.get_row(i)
        total_d = np.sum(demands)
        _, pred = __shortest_path(costs, forward_star, edge_map, source=i, targets=destinations, node_order=node_order)
        path_costs, paths = __pred_to_epath(pred, i, destinations, edge_map, costs)
        for path, path_flow, path_cost in zip(paths, demands, path_costs):
            ssp_costs[counter] = path_cost
            counter += 1
            for link_id in path:
                flows[link_id] += path_flow
    return ssp_costs, flows


def __demand_loss(forward_star, i, origin_weight, link_flows, edge_map):
    # this is merely here for debugging purposes..
    loaded = 0
    for j in forward_star[i]:
        print('linkid')
        print(edge_map[(i, j)])
        print(link_flows[edge_map[(i, j)]])
        loaded += link_flows[edge_map[(i, j)]]
        print(loaded)
        print(origin_weight)
    if int(loaded) < int(origin_weight):
        print('origin ')
        print(i)
        print('checked out with a demand of')
        print(loaded)
        print(origin_weight)
        return True
    else:
        return False


@njit
def __topological_order(distances):
    '''

    Parameters
    ----------
    distances : dictionary of distances keyed by node ids

    Returns
    -------

    '''
    topological_order = np.empty(len(distances), dtype=np.uint32)
    for counter, (node, dist) in enumerate(sorted(distances.items(), key=lambda x: x[1])):
        topological_order[counter] = node
    return topological_order


@njit
def __valid_edges(edge_map, label):
    '''

    Parameters
    ----------
    edge_map : dictionary keyed by (i,j) with edge index as value
    distances : dictionary of distances keyed by node ids

    Returns
    -------

    '''
    edges = List()
    for (i, j) in edge_map:
        if label[j] > label[i]:
            edges.append((i, j))
    return edges
