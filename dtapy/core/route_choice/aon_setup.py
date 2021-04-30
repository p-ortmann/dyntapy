#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from dtapy.core.route_choice.dynamic_dijkstra import dijkstra
import numpy as np
from dtapy.core.route_choice.aon_cls import RouteChoiceState
from dtapy.core.supply import Network
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime
from dtapy.datastructures.csr import F32CSRMatrix, csr_prep
from numba.typed import List
from dtapy.utilities import _log
from dtapy.core.route_choice.aon import update_arrival_maps, get_turning_fractions, get_source_connector_choice
from numba import njit


@njit(cache=True)
def init_arrival_maps(costs, in_links, destinations, step_size, tot_time_steps, tot_nodes, centroids):
    is_centroid = np.full(tot_nodes, False)
    for centroid in centroids:
        is_centroid[centroid] = True
    arrival_map = np.empty((len(destinations), tot_time_steps, tot_nodes), dtype=np.float32)

    for _id, destination in enumerate(destinations):
        arrival_map[_id, 0, :] = dijkstra(costs[0, :], in_links, destination, tot_nodes, is_centroid)
        for t in range(1, tot_time_steps):
            arrival_map[_id, t, :] = arrival_map[_id, 0,
                                     :] + t * step_size  # init of all time steps with free flow vals
    return arrival_map


# @njit(cache=True)
def setup_aon(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand):
    free_flow_costs = network.links.length / network.links.v0
    costs= np.empty((time.tot_time_steps, network.tot_links), dtype=np.float32)
    for t in range(time.tot_time_steps):
        costs[t, :] = free_flow_costs

    arrival_maps = init_arrival_maps(costs, network.nodes.in_links,
                                     dynamic_demand.all_active_destinations, time.step_size, time.tot_time_steps,
                                     network.tot_nodes, dynamic_demand.all_centroids)

    source_connector_choice = List()
    last_source_connector = np.max(np.argwhere(network.links.link_type == 1))  # highest link_id of source connectors
    for t in dynamic_demand.loading_time_steps:
        _id = 0
        index_array = np.empty((network.tot_connectors * dynamic_demand.tot_active_destinations,
                                np.uint32(2)), dtype=np.uint32)
        val = np.zeros(network.tot_connectors * dynamic_demand.tot_active_destinations, dtype=np.float32)
        demand = dynamic_demand.get_demand(t)
        for origin in demand.origins:
            for destination in demand.to_destinations.get_nnz(origin):
                destination_id = np.argwhere(dynamic_demand.all_active_destinations == destination)[0, 0]
                for connector in network.nodes.out_links.get_nnz(origin):
                    index_array[_id] = [connector, destination_id]
                    _id += 1
        index_array = index_array[:_id].copy()
        val = val[:_id].copy()
        val, col, row = csr_prep(index_array, val,
                                 shape=(last_source_connector + 1, dynamic_demand.tot_active_destinations + 1))
        source_connector_choice.append(
            F32CSRMatrix(val, col, row))
    _log('Calculating initial turning fractions', to_console=True)
    turning_fractions = get_turning_fractions(dynamic_demand, network, time, arrival_maps, costs)
    _log('Calculating initial source connector choice', to_console=True)
    connector_choice = get_source_connector_choice(network, source_connector_choice, arrival_maps, dynamic_demand)
    return RouteChoiceState(costs, arrival_maps, turning_fractions, connector_choice)
