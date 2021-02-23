#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dtapy.assignment import Assignment
from dtapy.core.route_choice.aon_cls import AONState
import numpy as np
from dtapy.parameters import parameters
from numba import prange

route_choice_delta = parameters.route_choice.delta_cost
route_choice_agg = parameters.route_choice.aggregation


# TODO: add generic results object
@njit
def update_arrival_maps(assignment: Assignment, state: AONState):
    tot_time_steps = assignment.time.route_choice.tot_time_steps
    from_node = assignment.network.links.from_node
    to_node = assignment.network.links.to_node
    out_links = assignment.network.nodes.out_links
    in_links = assignment.network.nodes.in_links
    all_destinations = assignment.dynamic_demand.all_destinations
    state.prev_costs = state.cur_costs
    state.cur_costs = assignment.results.costs
    delta_costs = np.abs(state.cur_costs - state.prev_costs)
    nodes_2_update = np.full(assignment.network.tot_nodes, np.iinfo(np.uint32).max, dtype=np.uint32)
    next_nodes_2_update = np.full(assignment.network.tot_nodes,np.iinfo(np.uint32).max, dtype=np.uint32)  # nodes to be
    # activated for earlier time steps
    arrival_maps = state.arrival_maps
    step_size = assignment.time.route_choice.step_size
    link_time = np.floor(state.cur_costs / step_size)
    interpolation_frac = state.cur_costs / step_size - link_time
    # TODO: revisit structuring of the travel time arrays
    # could be worth while to copy and reverse the order depending on where you're at in these loops ..
    # TODO: replace node2update  lists with fixed size arrays and pointers
    # the following implementation closely follows the solution presented in
    # Himpe, Willem. "Integrated Algorithms for Repeated Dynamic Traffic Assignments The Iterative
    # Link Transmission Model with Equilibrium Assignment Procedure."(2016).
    # refer to page 48, algorithm 6 for details.
    next_nodes_2_update_pointer = 0  # first elements in the array that are not nodes
    tot_nodes_2_update = 0  # number of valid entries in nodes_2_update
    tot_next_nodes_2_update = 0
    tot_active_nodes = 0  # number of nodes in this time step that still need to be considered
    for destination in all_destinations:
        for t in range(tot_time_steps, 0, -1):
            nodes_2_update[:tot_next_nodes_2_update] = next_nodes_2_update[:tot_next_nodes_2_update].copy()
            tot_nodes_2_update = tot_next_nodes_2_update.copy()
            for link, delta in np.ndenumerate(delta_costs[t, :]):
                # find all links with changed travel times and add their tail nodes
                # to the list of nodes to be updated
                if delta > route_choice_delta:
                    node = from_node[link]
                    nodes_2_update[tot_nodes_2_update] = node
                    tot_nodes_2_update += 1
            unique_nodes = np.unique(nodes_2_update[:tot_nodes_2_update])
            tot_nodes_2_update = unique_nodes.size
            nodes_2_update[:tot_nodes_2_update] = unique_nodes
            tot_active_nodes = tot_nodes_2_update.size
            while tot_active_nodes > 0:
                # going through all the nodes that need updating for the current time step
                # note that nodes_2_update changes dynamically as we traverse the graph ..
                # finding the node with the minimal arrival time to the destination is meant
                # to reduce the total nodes being added to the nodes_2_update list

                # TODO: explore some other designs here -  like priority queue
                # not straight forward to do as the distance labels are dynamically changing inside a single time step.

                min_dist = np.inf
                min_idx = -1
                for idx, node in enumerate(nodes_2_update[:tot_nodes_2_update]):
                    try:
                        if arrival_maps[destination, t, node] < min_dist:
                            min_idx = idx
                            min_dist = arrival_maps[destination, t, node]
                    except Exception:
                        # node ==  np.iinfo(np.uint32)
                        # IndexError .. already queried.
                        continue
                node = nodes_2_update[min_idx]
                nodes_2_update[min_idx] = np.iinfo(np.uint32)  # no longer considered
                tot_active_nodes = tot_active_nodes - 1
                new_dist = np.inf
                for link in out_links.get_nnz(node):

                    if t + link_time > tot_time_steps:
                        dist = arrival_maps[destination, tot_time_steps, to_node[link]] + state.cur_costs[t, link] \
                               - (tot_time_steps - t) * step_size
                    else:
                        dist = (1 - interpolation_frac) * arrival_maps[
                            destination, t + link_time, to_node[link]] + interpolation_frac * arrival_maps[
                                   destination, t + link_time + 1, to_node[link]]
                    if dist < new_dist:
                        new_dist = dist
                if np.abs(new_dist - arrival_maps[destination, t, node]) > route_choice_delta:
                    # new arrival time found
                    arrival_maps[destination, t, node] = new_dist
                    if node > assignment.dynamic_demand.tot_centroids:
                        # only adds the in_links if it's not a centroid
                        # the first nodes are centroids, see labelling in assignment.py
                        for link in in_links.get_nnz(node):
                            nodes_2_update[tot_nodes_2_update]=from_node[link]
                            next_nodes_2_update[tot_next_nodes_2_update]=from_node[link]
                            tot_next_nodes_2_update += 1
                            tot_nodes_2_update += 1
                            tot_active_nodes += 1


# TODO: test the @njit(parallel=True) option here
@njit(parallel=True)
def calc_turning_fractions(assignment: Assignment, state: AONState, departure_time_offset=route_choice_agg):
    """

    Parameters
    ----------
    assignment : Assignment, see def
    state : AONState, see def
    departure_time_offset : float32 in [0,1] , indicates which departure time to consider
     in between two adjacent time intervals
    0 indicates the very first vehicle is used to predict the choices of all in the interval,
    0.5 the middle, and consequently 1 the last


    Returns
    -------

    """
    # calculation for the experienced travel times
    update_arrival_maps(assignment, state)
    arrival_maps = state.arrival_maps
    step_size = assignment.time.route_choice.step_size
    next_link = np.int32(-1)
    next_node = np.int32(-1)
    turning_fractions = state.turning_fractions
    # starting point tomorrow - all that needs to be done is to query into the future for the smallest label in the current time step!
    for idx in prange(assignment.dynamic_demand.all_destinations.size):
        destination = assignment.dynamic_demand.all_destinations[idx]
        dists = state.arrival_maps[destination, :, :]
        for t in assignment.time.route_choice.tot_time_steps:
            for node in assignment.network.nodes:
                next_node = node.copy()
                start_time = t + departure_time_offset
                min_dist = np.inf
                for link, to_node in zip(assignment.network.nodes.out_links.get_nnz(next_node),
                                         assignment.network.nodes.out_links.get_row(next_node)):
                    link_time = np.floor(start_time+state.cur_costs[t,link])
                    interpolation_fraction = start_time+state.cur_costs[t,link] - link_time
                    dist = (1 - interpolation_fraction) * arrival_maps[
                        destination, t + link_time, to_node] + interpolation_fraction * arrival_maps[
                               destination, t + link_time + 1, to_node]
                    if dist < min_dist:
                        next_link = link
                for turn in assignment.network.turns.to_link[next_link]:
                    turning_fractions[destination, t, turn] = 1