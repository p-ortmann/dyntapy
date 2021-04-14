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
from dtapy.core.route_choice.aon_cls import AONState
from dtapy.core.supply import Network
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime
from dtapy.datastructures.csr import F32CSRMatrix, csr_prep
from numba.typed import List
from numba import njit

#@njit(cache=True)
def _init_arrival_maps(costs,in_links, destinations, step_size, tot_time_steps, tot_nodes, centroids):
    is_centroid = np.full(tot_nodes, False)
    for centroid in centroids:
        is_centroid[centroid] = True
    arrival_map = np.empty((len(destinations), tot_time_steps, tot_nodes), dtype=np.float32)

    for _id, destination in enumerate(destinations):
        arrival_map[_id, 0, :] = dijkstra(costs[0, :],in_links, destination, tot_nodes, is_centroid)
        for t in range(1, tot_time_steps):
            arrival_map[_id, t, :] = arrival_map[_id, 0,
                                     :] + t * step_size * 3600  # init of all time steps with free flow vals
    return arrival_map


#@njit(cache=True)
def setup_aon(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand):
    costs = network.links.length / network.links.v0
    costs = 3600 * costs
    step_size = time.step_size
    cur_costs = np.empty((time.tot_time_steps, network.tot_links), dtype=np.float32)
    for t in range(time.tot_time_steps):
        cur_costs[t, :] = costs
    prev_costs = np.copy(cur_costs)
    tot_time_steps = time.tot_time_steps
    tot_turns = network.tot_turns
    tot_destinations = dynamic_demand.tot_active_destinations
    for destination in dynamic_demand.all_active_destinations:
        connectors = network.nodes.in_links.get_nnz(destination)
        for c in connectors:
            prev_costs[:, c] = np.inf
            # triggering recomputations along all paths towards the destination

    arrival_maps = _init_arrival_maps(cur_costs,network.nodes.in_links,
                                      dynamic_demand.all_active_destinations, time.step_size, time.tot_time_steps,
                                      network.tot_nodes, dynamic_demand.all_centroids)
    turning_fractions = np.zeros((tot_destinations, tot_time_steps, tot_turns), dtype=np.float32)
    turning_fractions = np.zeros((tot_destinations, tot_time_steps, tot_turns), dtype=np.float32)
    link_time = np.floor(cur_costs / step_size)
    interpolation_frac = cur_costs / step_size - link_time

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
    return AONState(cur_costs, prev_costs, arrival_maps, turning_fractions,
                    interpolation_frac, link_time, source_connector_choice)
