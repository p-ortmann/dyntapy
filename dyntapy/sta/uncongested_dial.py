#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# the
from math import exp

from numba import njit
import numpy as np

from dyntapy.demand import InternalStaticDemand
from dyntapy.graph_utils import (
    dijkstra_all,
    make_in_links,
    make_out_links,
)
from dyntapy.settings import parameters
from dyntapy.supply import Network


# since this works with topological orders the full path set is not considered.
def sun(network: Network, demand: InternalStaticDemand):
    theta = parameters.static_assignment.logit_theta
    ff_times = network.links.length / network.links.free_speed
    topological_orders, links_in_bush, distances = generate_bushes(
        ff_times,
        network.links.from_node,
        network.links.to_node,
        network.nodes.out_links,
        demand,
        network.tot_links,
        network.nodes.is_centroid,
    )
    return ff_times, *load_all_bushes(
        demand,
        topological_orders,
        links_in_bush,
        distances,
        ff_times,
        theta,
        network,
    )


def load_all_bushes(
    demand,
    topological_orders,
    links_in_bush,
    distances,
    costs,
    theta,
    network,
):
    tot_links = network.tot_links
    from_nodes = network.links.from_node
    to_nodes = network.links.to_node
    tot_nodes = network.tot_nodes
    tot_origins = demand.to_destinations.get_nnz_rows().size
    bush_flows = np.zeros((tot_origins, tot_links))
    for bush_id, bush in enumerate(demand.to_destinations.get_nnz_rows()):
        bush_out_links = make_out_links(
            links_in_bush[bush_id], from_nodes, to_nodes, tot_nodes
        )
        bush_in_links = make_in_links(
            links_in_bush[bush_id], from_nodes, to_nodes, tot_nodes
        )
        destinations = demand.to_destinations.get_nnz(bush)
        demands = demand.to_destinations.get_row(bush)
        load_bush(
            bush,
            costs,
            destinations,
            demands,
            topological_orders[bush_id],
            links_in_bush[bush_id],
            distances[bush_id],
            bush_flows[bush_id],
            theta,
            bush_out_links,
            bush_in_links,
            tot_nodes,
            tot_links,
            from_nodes,
            to_nodes,
        )

    return np.sum(bush_flows, axis=0), bush_flows


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


# @njit()
def load_bush(
    origin,
    costs,
    destinations,
    demands,
    topological_order,
    links_in_bush,
    distances,
    bush_flows,
    theta,
    out_links,
    in_links,
    tot_nodes,
    tot_links,
    from_nodes,
    to_nodes,
):
    link_weights, node_weights = set_labels(
        origin,
        out_links,
        in_links,
        links_in_bush,
        distances,
        topological_order,
        from_nodes,
        to_nodes,
        costs,
        theta,
    )
    node_flows = np.zeros(tot_nodes, dtype=np.float32)
    link_flows = np.zeros(tot_links, dtype=np.float32)
    # print('just before loading')
    for j in topological_order[::-1]:
        destination_demand = 0.0
        passing_demand = 0.0
        for index, _j in enumerate(destinations):
            if j == _j:
                destination_demand = float(demands[index])
                break
        for i, link in out_links[j]:
            passing_demand += link_flows[link]
        node_flows[j] = destination_demand + passing_demand
        for i, link in in_links[j]:
            assert node_weights[j] > 0
            link_flows[link] = float(
                node_flows[j] * link_weights[link] / node_weights[j]
            )

    for link, flow in enumerate(link_flows):
        bush_flows[link] += link_flows[link]


# @njit()
def set_labels(
    origin,
    out_links,
    in_links,
    links_in_bush,
    distances,
    topological_order,
    from_nodes,
    to_nodes,
    costs,
    theta,
):
    link_likelihood = np.zeros(costs.size, dtype=np.float32)
    node_weights = np.zeros(distances.size, dtype=np.float32)
    node_weights[origin] = 1.0
    link_weights = np.zeros(costs.size, dtype=np.float32)
    for link, (in_bush, i, j) in enumerate(zip(links_in_bush, from_nodes, to_nodes)):
        # larger theta leads to AON behavior
        link_likelihood[link] = exp(
            theta * (-costs[link] - distances[i] + distances[j])
        )
    for i in topological_order:
        if i != origin:
            node_weights[i] = 0.0
            for j, link in in_links[i]:
                node_weights[i] += link_weights[link]
            assert node_weights[i] > 0.00001
        for j, link in out_links[i]:
            link_weights[link] = link_likelihood[link] * node_weights[i]
    return link_weights, node_weights
