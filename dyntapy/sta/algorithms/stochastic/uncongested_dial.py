#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# the
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.settings import static_parameters
from numba import njit, jit
from numba.typed import Dict, List
from dyntapy.sta.algorithms.graph_utils import __shortest_path, __pred_to_epath2, make_out_links, make_in_links
from dyntapy.sta.algorithms.helper_funcs import __valid_edges, __topological_order
from math import exp
from collections import defaultdict
import numpy as np


def uncongested_stochastic_assignment(obj: StaticAssignment):
    theta = static_parameters.assignment.logit_theta
    # print(f'my theta is {theta}')
    flows = np.zeros(obj.tot_links)
    topological_orders, edges, L, largest_destination_labels = generate_bushes(obj.link_ff_times, obj.edge_map,
                                                                               obj.out_links,
                                                                               obj.demand, obj.tot_nodes)
    return load_all_bushes(obj, topological_orders, edges, L, largest_destination_labels, obj.link_ff_times, theta)


def load_all_bushes(obj, topological_orders, edges, L, largest_destination_labels, costs, theta):
    flows = np.zeros(obj.tot_links)
    for bush in obj.demand.to_destinations.get_nnz_rows():
        forward_star = make_out_links(edges[bush], len(topological_orders[bush]))
        backward_star = make_in_links(edges[bush], len(topological_orders[bush]))
        destinations = obj.demand.to_destinations.get_nnz(bush)
        demands = obj.demand.to_destinations.get_row(bush)
        flows = load_bush(bush, obj.edge_map, costs, destinations, demands,
                          topological_orders[bush],
                          edges[bush], L[bush], flows, largest_destination_labels[bush], theta, forward_star,
                          backward_star)
    return flows


# @njit
def generate_bushes(link_ff_times, edge_map, global_forward_star, demand, node_order):
    topological_orders = Dict()
    edges = Dict()
    largest_destination_labels = Dict()
    L = Dict()
    # forward_stars= defaultdict()
    # backward_stars=defaultdict()
    assert demand.to_destinations.get_nnz_rows().size>0
    for bush in demand.to_destinations.get_nnz_rows():
        destinations = demand.to_destinations.get_nnz(bush)
        L[bush], pred = __shortest_path(costs=link_ff_times, forward_star=global_forward_star, edge_map=edge_map,
                                        source=bush,
                                        targets=np.empty(0), node_order=node_order)
        topological_orders[bush] = __topological_order(L[bush])
        label = Dict()
        for j in topological_orders[bush]:
            label[topological_orders[bush][j]] = j
        edges[bush] = __valid_edges(edge_map, label)
        largest_destination_labels[bush] = 0
        # forward_stars[bush] = make_out_links(edges[bush], len(pred))
        # backward_stars[bush]= make_in_links(edges[bush], len(pred))
        for destination in destinations:
            largest_destination_labels[bush] = max(label[destination], largest_destination_labels[bush])
        # print('created bush')
    return topological_orders, edges, L, largest_destination_labels


# @njit()
def load_bush(origin, edge_map, costs, destinations, demands, topological_order, bush_edges, L, flows,
              largest_destination, theta, forward_star, backward_star):
    edge_weights, node_weights = set_labels(origin, forward_star, backward_star, bush_edges, L, topological_order,
                                            largest_destination, edge_map, costs, theta)
    node_flows = Dict()
    edge_flows = Dict()
    # print('just before loading')
    for j in topological_order[largest_destination::-1]:
        destination_demand = 0.0
        passing_demand = 0.0
        for index, _j in enumerate(destinations):
            if j == _j:
                destination_demand = float(demands[index])
        for i in forward_star[j]:
            try:
                passing_demand += edge_flows[(j, i)]
            except Exception:
                # edge not loaded..
                continue
        node_flows[j] = destination_demand + passing_demand
        for i in backward_star[j]:
            if node_weights[j] == 0:
                print(j)
                print(backward_star[j])
            assert node_weights[j] > 0
            edge_flows[(i, j)] = float(node_flows[j] * edge_weights[(i, j)] / node_weights[j])

    for edge in edge_flows.keys():
        (i, j) = edge
        edge = edge_map[(i, j)]
        flows[edge] += edge_flows[(i, j)]
    return flows


# @njit()
def set_labels(origin, forward_star, backward_star, bush_edges, L, topological_order, largest_destination,
               edge_map, costs, theta):
    edge_likelihood = Dict()
    node_weights = Dict()
    node_weights[origin] = 1.0
    edge_weights = Dict()
    for edge in bush_edges:
        (i, j) = edge
        edge_likelihood[edge] = exp(theta * (-costs[edge_map[edge]] - L[i] + L[j]))
    for i in topological_order[:largest_destination + 1]:
        if i != origin:
            node_weights[i] = 0.0
            for j in backward_star[i]:
                node_weights[i] += edge_weights[(j, i)]
            if node_weights[i] == 0.0:
                print('here')
                print(i)
                print(i)
        for j in forward_star[i]:
            edge_weights[(i, j)] = edge_likelihood[(i, j)] * node_weights[i]
    return edge_weights, node_weights
