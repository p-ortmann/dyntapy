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
from dtapy.assignment import Assignment
from dtapy.core.route_choice.aon_cls import AONState
from numba import njit


@njit()
def _init_arrival_maps(costs, out_links, destinations, step_size, tot_time_steps, tot_nodes):
    arrival_map = np.empty((len(destinations), tot_time_steps, tot_nodes), dtype=np.float32)
    for _id, destination in enumerate(destinations):
        dijkstra(costs, out_links, destination, arrival_map[_id, 0, :])
        for t in range(1, tot_time_steps):
            arrival_map[_id, t, :] = arrival_map[_id, 0,
                                     :] + t * step_size  # init of all time steps with free flow vals
    return arrival_map


@njit()
def setup_aon(assignment: Assignment):
    cur_costs = assignment.network.links.length / assignment.network.links.v0
    step_size= assignment.time.step_size
    prev_costs = np.copy(cur_costs)
    tot_time_steps  = assignment.time.tot_time_steps
    tot_turns = assignment.network.tot_turns
    tot_destinations =  assignment.network.tot_turns
    for destination in assignment.dynamic_demand.all_destinations:
        connectors = assignment.network.nodes.in_links.get_nnz(destination)
        for c in connectors:
            prev_costs[c] = np.inf
            # triggering recomputations along all paths towards the destination

    arrival_maps = _init_arrival_maps(cur_costs, assignment.network.nodes.out_links,
                                      assignment.dynamic_demand.all_destinations, assignment.time.step_size,
                                      assignment.network.tot_nodes)
    turning_fractions = np.zeros((tot_time_steps, tot_destinations, tot_turns), dtype=np.float32)
    link_time = np.floor(cur_costs / step_size)
    interpolation_frac = cur_costs / step_size - link_time
    return AONState(cur_costs, prev_costs, arrival_maps, turning_fractions, interpolation_frac, link_time)
