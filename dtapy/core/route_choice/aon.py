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
from dtapy.settings import parameters
import numpy as np
from numba import prange
from numba.typed import List
from heapq import heappush, heappop
from dtapy.utilities import _log
from dtapy.datastructures.csr import F32CSRMatrix, UI32CSRMatrix

route_choice_delta = parameters.route_choice.delta_cost
route_choice_agg = parameters.route_choice.aggregation


# TODO: test function for arrival map topological order

# @njit(cache=True, parallel=True)
def update_arrival_maps(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand, arrival_maps,
                        old_costs, new_costs):
    tot_time_steps = time.tot_time_steps
    from_node = network.links.from_node
    to_node = network.links.to_node
    out_links = network.nodes.out_links
    in_links = network.nodes.in_links
    all_destinations = dynamic_demand.all_active_destinations
    delta_costs = np.abs(new_costs - old_costs)
    next_nodes_2_update = np.full(network.tot_nodes, False, dtype=np.bool_)  # nodes to be
    # activated for earlier time steps
    step_size = time.step_size
    np.floor_divide(new_costs, step_size)
    link_time = np.floor_divide(new_costs, step_size)
    interpolation_frac = np.divide(new_costs, step_size) - link_time
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
            _log('building map for destination ' + str(destination) + ' , now in time step ' + str(t), to_console=True)
            nodes_2_update = next_nodes_2_update.copy()
            for link, delta in np.ndenumerate(delta_costs[t, :]):
                # find all links with changed travel times and add their tail nodes
                # to the list of nodes to be updated
                if delta > route_choice_delta:
                    node = from_node[link]
                    if node != dynamic_demand.all_active_destinations[destination]:
                        nodes_2_update[node] = True
            while np.any(nodes_2_update == True):
                # _log('currently active nodes: ' + str(np.argwhere(nodes_2_update == True)))
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
                # _log('deactivated node ' + str(min_node))
                new_dist = np.inf
                for out_node, link in zip(out_links.get_row(min_node), out_links.get_nnz(min_node)):
                    if out_node != dynamic_demand.all_active_destinations[
                        destination] and out_node < dynamic_demand.tot_centroids:
                        # centroids cannot be part of a path and not be a terminal node
                        continue
                    else:
                        if t + np.uint32(link_time[t, link]) >= tot_time_steps - 1:
                            dist = arrival_maps[destination, tot_time_steps - 1, out_node] + new_costs[t, link]
                            - (tot_time_steps - 1 - t) * step_size
                        else:
                            dist = (1 - interpolation_frac[t, link]) * arrival_maps[
                                destination, t + np.uint32(link_time[t, link]), out_node] + interpolation_frac[
                                       t, link] * arrival_maps[
                                       destination, t + np.uint32(link_time[t, link]) + 1, out_node]
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
                            if from_node[link] != dynamic_demand.all_active_destinations[destination]:
                                # _log('activated node ' + str(from_node[link]))
                                nodes_2_update[from_node[link]] = True
                                next_nodes_2_update[from_node[link]] = True


# TODO: test the @njit(parallel=True) option here
# @njit(cache=True)
def get_turning_fractions(dynamic_demand: InternalDynamicDemand, network: Network, time: SimulationTime, arrival_maps,
                          new_costs, departure_time_offset=route_choice_agg):
    """
    calculates turning fractions taking into account closed turns.
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
    # _log('calculating arrival maps ')

    step_size = time.step_size
    my_heap = []
    link_has_active_out_turn = np.full(np.max(network.nodes.tot_in_links), True)
    turning_fractions = np.zeros((dynamic_demand.tot_active_destinations, time.tot_time_steps, network.tot_turns),
                                 dtype=np.float32)
    min_dist=np.inf
    min_turn=-1
    for dest_idx in range(dynamic_demand.all_active_destinations.size):
        dest_link_id=dynamic_demand.all_active_destination_links[dest_idx]
        # print(f'destination {dynamic_demand.all_active_destinations[dest_idx]}')
        for t in range(time.tot_time_steps):
            for link in range(network.tot_links):
                min_dist=np.inf
                min_turn = -1
                for out_turn, to_link in zip(network.links.out_turns.get_nnz(link),
                                             network.links.out_turns.get_row(link)):
                    turn_time = np.floor(departure_time_offset + new_costs[t, out_turn] / step_size)
                    if t + np.uint32(turn_time) < time.tot_time_steps - 1:
                        interpolation_fraction = departure_time_offset + new_costs[
                            t, out_turn] / step_size - turn_time
                        dist = (1 - interpolation_fraction) * arrival_maps[
                            dest_idx, t + np.uint32(turn_time), to_link] + interpolation_fraction * arrival_maps[
                                   dest_idx, t + np.uint32(turn_time) + 1, to_link]
                    else:
                        dist = arrival_maps[dest_idx, time.tot_time_steps - 1, to_link] + new_costs[
                            t, out_turn]
                    if dist<=min_dist:
                        min_turn=out_turn
                        min_dist=dist
                turning_fractions[dest_idx, t, min_turn] = 1
                            # this does not actually assign any turning fraction to an in_link that does NOT
                            # have a turn that leads to next_link
    return turning_fractions


#@njit(parallel=True)
def link_to_turn_costs(link_costs: np.ndarray,out_links: UI32CSRMatrix, in_links: UI32CSRMatrix,
                       out_turns: UI32CSRMatrix, in_turns: UI32CSRMatrix, tot_turns):
    # TODO: testing of this function
    """
    calculates turn from link costs assuming no turn delays
    Parameters
    ----------
    out_links : csr, node x links
    link_costs : array, tot_time_steps x tot_links
    out_turns : csr, link x turns

    Returns
    -------
    turn_costs as an array, tot_time_steps x turns
    """
    tot_time_steps = link_costs.shape[0]
    turn_costs = np.zeros((tot_time_steps, tot_turns), dtype=np.float32)
    for node in prange(out_links.get_nnz_rows().size):
        # turn and link labelling follows the node labelling
        # turns with the same via node are labelled consecutively
        # the same is usually true for the outgoing links of a node (if it's not a connector)
        for link in out_links.get_nnz(node):
            for turn in in_turns.get_nnz(link):
                turn_costs[:, turn] += link_costs[:, link]
        for link in in_links.get_nnz(node):  # this is more expensive since the in_links are not labelled consecutively
            for turn in out_turns.get_nnz(link):
                turn_costs[:, turn] += link_costs[:, link]
    return turn_costs


# @njit(cache=True)
def get_source_connector_choice(network: Network, connector_choice: F32CSRMatrix, arrival_maps,
                                dynamic_demand: InternalDynamicDemand):
    """
    replaces the values in the CSRMatrices in connector choice with the current values with the given arrival maps
    and returns the result.
    Connector choice is passed to get the sparsity structure.
    Parameters
    ----------
    network : Network
    connector_choice : List of CSRMatrices, as defined in datastructures
    arrival_maps : 3D array, tot_active_destinations x time.tot_time_steps x tot_nodes
    dynamic_demand : InternalDynamicDemand

    Returns
    -------

    """

    for t_id, t in enumerate(dynamic_demand.loading_time_steps):
        demand = dynamic_demand.get_demand(t)
        connector_choice[t_id].values = np.zeros_like(connector_choice[0].values)  # initializing with zeros
        for origin in demand.origins:
            for d_id, destination in enumerate(demand.to_destinations.get_nnz(origin)):
                dist = np.inf
                min_connector = -1
                for node, connector in zip(network.nodes.out_links.get_row(origin),
                                           network.nodes.out_links.get_nnz(origin)):
                    if arrival_maps[d_id, t, node] < dist:
                        dist = arrival_maps[d_id, t, node]
                        min_connector = connector
                connector_choice[t_id].get_row(min_connector)[d_id] = 1.0
    return connector_choice


@njit(cache=True)
def update_source_connector_choice(network: Network, connector_choice: F32CSRMatrix, arrival_maps,
                                   dynamic_demand: InternalDynamicDemand):
    """
    replaces the values in the CSRMatrices in connector choice with the current values with the given arrival maps
    and returns the result.
    Connector choice is passed to get the sparsity structure.
    Parameters
    ----------
    network : Network
    connector_choice : List of CSRMatrices, as defined in datastructures
    arrival_maps : 3D array, tot_active_destinations x time.tot_time_steps x tot_nodes
    dynamic_demand : InternalDynamicDemand

    Returns
    -------

    """
    for t_id, t in enumerate(dynamic_demand.loading_time_steps):
        demand = dynamic_demand.get_demand(t)
        connector_choice[t_id].values = np.zeros_like(connector_choice[0].values)  # initializing with zeros
        for origin in demand.origins:
            for d_id, destination in enumerate(demand.to_destinations.get_nnz(origin)):
                dist = np.inf
                min_link = -1
                for node, link in zip(network.nodes.out_links.get_row(origin), network.nodes.out_links.get_nnz(origin)):
                    if arrival_maps[d_id, t, node] < dist:
                        dist = arrival_maps[d_id, t, node]
                        min_link = link
                connector_choice[t_id].get_row(min_link)[d_id] = 1.0
    return connector_choice
