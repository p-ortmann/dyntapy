#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dtapy.core.network_loading.link_models.i_ltm_cls import ILTMState
from dtapy.core.network_objects_cls import SimulationTime, Network
from dtapy.assignment import Assignment
from dtapy.parameters import route_choice_dt, route_choice_agg
from dtapy.core.route_choice.dynamic_dijkstra import dijkstra
from dtapy.core.route_choice.aon_cls import AONState
import numpy as np
from dtapy.parameters import route_choice_delta
from heapq import heappop, heappush
from numba.typed import List


# TODO: add generic results object
@njit
def update_arrival_maps(assignment: Assignment, state: AONState):
    tot_time_steps = assignment.time.tot_time_steps
    from_node = assignment.network.links.from_node
    to_node = assignment.network.links.to_node
    out_links = assignment.network.nodes.out_links
    in_links = assignment.network.nodes.in_links
    all_destinations = assignment.dynamic_demand.all_destinations
    state.prev_costs = state.cur_costs
    state.cur_costs = assignment.results.costs
    delta_costs = np.abs(state.cur_costs - state.prev_costs)
    nodes_2_update = List() # list of nodes that have to be updated for the current time step
    next_nodes_2_update = List() # for earlier time steps
    arrival_maps = state.arrival_maps
    step_size = assignment.time.step_size
    # TODO: revisit structuring of the travel time arrays
    # could be worth while to copy and reverse the order depending on where you're at in these loops ..

    # the following implementation closely follows the solution presented in
    # Himpe, Willem. "Integrated Algorithms for Repeated Dynamic Traffic Assignments The Iterative
    # Link Transmission Model with Equilibrium Assignment Procedure."(2016).
    # refer to page 48, algorithm 6 for details.
    for destination in all_destinations:
        for t in range(tot_time_steps, 0, -1):
            nodes_2_update = next_nodes_2_update
            for link, delta in np.ndenumerate(delta_costs[t, :]):
                # find all links with changed travel times and add their tail nodes
                # to the heap of nodes to be updated
                if delta > route_choice_delta:
                    node = from_node[link]
                    nodes_2_update.append(node)

            while len(nodes_2_update) > 0:
                # going through all the nodes that need updating for the current time step
                # note that nodes_2_update changes dynamically as we traverse the graph ..
                # finding the node with the minimal arrival time to the destination is meant
                # to reduce the total nodes being added to the nodes_2_update list

                # TODO: explore some other designs here -  like priority queue
                # not straight forward to do as the distance labels are dynamically changing inside a single time step.

                min_dist = np.inf
                min_idx = -1
                for idx, node in enumerate(nodes_2_update):
                    if arrival_maps[destination, t, node] < min_dist:
                        min_idx = idx
                        min_dist = arrival_maps[destination, t, node]
                node = nodes_2_update[min_idx]
                new_dist = np.inf
                for link in out_links.get_nnz(node):
                    link_time = np.floor(state.cur_costs[t, link] / step_size)
                    interpolation_frac = state.cur_costs[t, link] / step_size - link_time
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
                    for link in in_links.get_nnz(node):
                        nodes_2_update.append(from_node[link])
                        next_nodes_2_update.append(from_node[link])
