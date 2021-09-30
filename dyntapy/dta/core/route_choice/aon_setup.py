#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from dyntapy.dta.core.route_choice.dynamic_dijkstra import dijkstra
import numpy as np
from dyntapy.dta.core.route_choice.deterministic import RouteChoiceState
from dyntapy.dta.core.supply import Network
from dyntapy.dta.core.demand import InternalDynamicDemand
from dyntapy.dta.core.time import SimulationTime
from dyntapy.datastructures.csr import F32CSRMatrix, csr_prep
from numba.typed import List
from dyntapy.utilities import _log
from dyntapy.dta.core.route_choice.aon import update_arrival_maps, get_turning_fractions, link_to_turn_costs
from numba import njit, uint32
from dyntapy.settings import dynamic_parameters
from dyntapy.dta.core.network_loading.link_models import i_ltm

restricted_turn_cost = dynamic_parameters.route_choice.restricted_turn_cost


@njit(cache=True)
def init_arrival_maps(costs, in_links, destinations, step_size, tot_time_steps, tot_nodes, centroids,
                      turn_restrictions):
    # works for node link and link turn graph representation
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


@njit(cache=True)
def setup_aon(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand):
    # in order to prevent excessive spillback and the connected, rather weird, equilibria we load in increments
    free_flow_costs = network.links.length / network.links.v0
    costs = np.empty((time.tot_time_steps, network.tot_links), dtype=np.float32)
    for t in range(time.tot_time_steps):
        costs[t, :] = free_flow_costs
    turn_costs \
        = link_to_turn_costs(costs, network.nodes.out_links,
                             network.links.in_turns, network.tot_turns, time,
                             np.empty((time.tot_time_steps, network.tot_turns)), use_turn_delays=False)
    turn_restrictions = np.full(network.tot_turns, False, np.bool_)
    for turn, (from_node, to_node) in enumerate(zip(network.turns.from_node, network.turns.to_node)):
        if from_node == to_node:
            turn_restrictions[turn] = True

    for turn in range(network.tot_turns):
        if turn_restrictions[turn]:
            turn_costs[:, turn] = restricted_turn_cost
    arrival_maps = init_arrival_maps(turn_costs, network.links.in_turns,
                                     dynamic_demand.all_active_destination_links, time.step_size, time.tot_time_steps,
                                     network.tot_links, List.empty_list(uint32),
                                     turn_restrictions)  # since there are no u-turns centroid routing is
    # prevented by default.
    turning_fractions = get_turning_fractions(dynamic_demand, network, time, arrival_maps, turn_costs)
    i_ltm(network, dynamic_demand, iltm_state, time, turning_fractions, k)
    return RouteChoiceState(costs, turn_costs, arrival_maps, turning_fractions, turn_restrictions)
