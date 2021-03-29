#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dtapy.core.supply import Network
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime
from dtapy.core.route_choice.aon_cls import AONState
import numpy as np
from dtapy.settings import parameters
from numba import prange
from dtapy.utilities import _log
route_choice_delta = parameters.route_choice.delta_cost
route_choice_agg = parameters.route_choice.aggregation


@njit(parallel=True)
def update_arrival_maps(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand, state: AONState):
    tot_time_steps = time.tot_time_steps
    from_node = network.links.from_node
    to_node = network.links.to_node
    out_links = network.nodes.out_links
    in_links = network.nodes.in_links
    all_destinations = dynamic_demand.all_active_destinations
    delta_costs = np.abs(state.cur_costs - state.prev_costs)
    next_nodes_2_update = np.full(network.tot_nodes, False, dtype=np.bool_)  # nodes to be
    # activated for earlier time steps
    arrival_maps = state.arrival_maps
    step_size = time.step_size
    np.floor_divide(state.cur_costs, step_size)
    link_time = np.floor_divide(state.cur_costs, step_size)
    interpolation_frac = np.divide(state.cur_costs, step_size) - link_time
    # TODO: revisit structuring of the travel time arrays
    # could be worth while to copy and reverse the order depending on where you're at in these loops ..
    # the following implementation closely follows the solution presented in
    # Himpe, Willem. "Integrated Algorithms for Repeated Dynamic Traffic Assignments The Iterative
    # Link Transmission Model with Equilibrium Assignment Procedure."(2016).
    # refer to page 48, algorithm 6 for details.
    tot_active_nodes = 0  # number of nodes in this time step that still need to be considered
    for destination in prange(all_destinations.size):
        _log(' processing new destination')
        next_nodes_2_update = np.full(network.tot_nodes, False, dtype=np.bool_)
        for t in range(tot_time_steps - 1, -1, -1):
            _log('building map for destination ' + str(destination) + ' , now in time step ' + str(t))
            nodes_2_update = next_nodes_2_update.copy()
            for link, delta in np.ndenumerate(delta_costs[t, :]):
                # find all links with changed travel times and add their tail nodes
                # to the list of nodes to be updated
                if delta > route_choice_delta:
                    node = from_node[link]
                    nodes_2_update[node] = True
            while np.any(nodes_2_update == True):
                #_log('currently active nodes: ' + str(np.argwhere(nodes_2_update == True)))
                # going through all the nodes that need updating for the current time step
                # note that nodes_2_update changes dynamically as we traverse the graph ..
                # finding the node with the minimal arrival time to the destination is meant
                # to reduce the total nodes being added to the nodes_2_update list

                # TODO: explore some other designs here -  like priority queue
                # not straight forward to do as the distance labels are dynamically changing inside a single time step.

                min_dist = np.inf
                min_node = -1
                for node, active in enumerate(nodes_2_update):
                    if active:
                        if arrival_maps[destination, t, node] < min_dist:
                            min_node = node
                            min_dist = arrival_maps[destination, t, node]
                nodes_2_update[min_node] = False  # no longer considered
                #_log('deactivated node ' + str(min_node))
                new_dist = np.inf
                for out_node, link in zip(out_links.get_row(min_node),out_links.get_nnz(min_node)):
                    if out_node < dynamic_demand.tot_centroids:
                        # centroids cannot be part of a path and not be a terminal node
                        continue
                    else:
                        if t + np.uint32(link_time[t, link]) >= tot_time_steps - 1:
                            dist = arrival_maps[destination, tot_time_steps - 1, to_node[link]] + state.cur_costs[t, link] \
                                   - (tot_time_steps - t) * step_size
                        else:
                            dist = (1 - interpolation_frac[t, link]) * arrival_maps[
                                destination, t + np.uint32(link_time[t, link]), to_node[link]] + interpolation_frac[
                                       t, link] * arrival_maps[
                                       destination, t + np.uint32(link_time[t, link]) + 1, to_node[link]]
                        # _log(f'distance to {min_node} via out_link node {to_node[link]} is {dist} ')
                        if dist < new_dist:
                            new_dist = dist
                # _log(f'result for node {min_node} written back? {np.abs(new_dist - arrival_maps[destination, t, min_node]) > route_choice_delta}')
                if np.abs(new_dist - arrival_maps[destination, t, min_node]) > route_choice_delta:
                    # new arrival time found
                    arrival_maps[destination, t, min_node] = new_dist
                    if min_node > dynamic_demand.tot_centroids:
                        # only adds the in_links if it's not a centroid
                        # the first nodes are centroids, see labelling in assignment.py
                        for link in in_links.get_nnz(min_node):
                            #_log('activated node ' + str(from_node[link]))
                            nodes_2_update[from_node[link]] = True
                            next_nodes_2_update[from_node[link]] = True


# TODO: test the @njit(parallel=True) option here
@njit(parallel=True)
def calc_turning_fractions(dynamic_demand: InternalDynamicDemand, network: Network, time: SimulationTime,
                           state: AONState, departure_time_offset=route_choice_agg):
    """

    Parameters
    ----------
    network : numba.experimental.jitclass.boxing.Network
    dynamic_demand : numba.experimental.jitclass.boxing.InternalDynamicDemand
    state : AONState, see def
    departure_time_offset : float32 in [0,1] , indicates which departure time to consider
     in between two adjacent time intervals
    0 indicates the very first vehicle is used to predict the choices of all in the interval,
    0.5 the middle, and consequently 1 the last


    Returns
    -------

    """
    # calculation for the experienced travel times
    _log('calculating arrival maps ')
    update_arrival_maps(network, time, dynamic_demand, state)
    _log('got past arrival maps')
    arrival_maps = state.arrival_maps
    step_size = time.step_size
    next_link = np.int32(-1)
    next_node = np.int32(-1)
    turning_fractions = state.turning_fractions
    for dest_idx in prange(dynamic_demand.all_active_destinations.size):
        dists = state.arrival_maps[dest_idx, :, :]
        # print(f'destination {dynamic_demand.all_active_destinations[dest_idx]}')
        for t in range(time.tot_time_steps):
            # print(f'time {t}')
            for node in range(network.tot_nodes):
                next_node = node
                start_time = t + departure_time_offset
                min_dist = np.inf
                for link, to_node in zip(network.nodes.out_links.get_nnz(next_node),
                                         network.nodes.out_links.get_row(next_node)):
                    link_time = np.floor(start_time + state.cur_costs[t, link])
                    if t + np.uint32(link_time) < time.tot_time_steps - 1:
                        interpolation_fraction = start_time + state.cur_costs[t, link] - link_time
                        dist = (1 - interpolation_fraction) * arrival_maps[
                            dest_idx, t + np.uint32(link_time), to_node] + interpolation_fraction * arrival_maps[
                                   dest_idx, t + np.uint32(link_time) + 1, to_node]
                    else:
                        dist = arrival_maps[dest_idx, time.tot_time_steps - 1, to_node]
                    if dist < min_dist:
                        next_link = link
                for turn in network.links.in_turns.get_row(next_link):
                    turning_fractions[dest_idx, t, turn] = 1


@njit
def calc_source_connector_choice(network: Network, state: AONState,
                                 dynamic_demand: InternalDynamicDemand):
    for t_id, t in enumerate(dynamic_demand.loading_time_steps):
        demand = dynamic_demand.get_demand(t)
        for origin in demand.origins:
            for _id, destination in enumerate(demand.to_destinations.get_nnz(origin)):
                dist = np.inf
                min_link = -1
                for node, link in zip(network.nodes.out_links.get_row(origin), network.nodes.out_links.get_nnz(origin)):
                    if state.arrival_maps[destination, t, node] < dist:
                        dist = state.arrival_maps[destination, t, node]
                        min_link = link
                state.connector_choice[t_id].get_row(min_link)[_id] = 1.0