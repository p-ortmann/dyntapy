#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dyntapy.dta.core.supply import Network
from dyntapy.dta.core.demand import InternalDynamicDemand
from dyntapy.dta.core.time import SimulationTime
from dyntapy.settings import dynamic_parameters
import numpy as np
from numba import prange
from dyntapy.utilities import _log
from dyntapy.datastructures.csr import F32CSRMatrix, UI32CSRMatrix

restricted_turn_cost = dynamic_parameters.route_choice.restricted_turn_cost
route_choice_delta = dynamic_parameters.route_choice.delta_cost
route_choice_agg = dynamic_parameters.route_choice.aggregation
use_turn_delays = dynamic_parameters.network_loading.use_turn_delays


# TODO: test function for arrival map topological order

@njit(cache=True, parallel=True)
def update_arrival_maps(network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand, arrival_maps,
                        old_costs, new_costs):
    tot_time_steps = time.tot_time_steps
    from_link = network.turns.from_link
    out_turns = network.links.out_turns
    in_turns = network.links.in_turns
    all_destinations = dynamic_demand.all_active_destinations
    delta_costs = np.abs(new_costs - old_costs)
    next_links_to_update = np.full(network.tot_nodes, False, dtype=np.bool_)  # nodes to be
    # activated for earlier time steps
    step_size = time.step_size
    np.floor_divide(new_costs, step_size)
    turn_time = np.floor_divide(new_costs, step_size)
    interpolation_frac = np.divide(new_costs, step_size) - turn_time
    # TODO: revisit structuring of the travel time arrays
    # could be worth while to copy and reverse the order depending on where you're at in these loops ..
    # the following implementation closely follows the solution presented in
    # Himpe, Willem. "Integrated Algorithms for Repeated Dynamic Traffic Assignments The Iterative
    # Link Transmission Model with Equilibrium Assignment Procedure."(2016).
    # refer to page 48, algorithm 6 for details.
    tot_active_nodes = 0  # number of nodes in this time step that still need to be considered
    for destination in prange(all_destinations.size):
        _log(' processing new destination')
        next_links_to_update = np.full(network.tot_links, False, dtype=np.bool_)
        for t in range(tot_time_steps - 1, -1, -1):
            _log('building map for destination ' + str(destination) + ' , now in time step ' + str(t), to_console=False)
            links_2_update = next_links_to_update.copy()
            for turn, delta in np.ndenumerate(delta_costs[t, :]):
                # find all links with changed travel times and add their tail nodes
                # to the list of nodes to be updated
                # u turn costs are set to infinity and do not change.
                if delta > route_choice_delta:
                    link = from_link[turn]
                    links_2_update[link] = True
            while np.any(links_2_update == True):
                # _log('currently active nodes: ' + str(np.argwhere(nodes_2_update == True)))
                # going through all the nodes that need updating for the current time step
                # note that nodes_2_update changes dynamically as we traverse the graph ..
                # finding the node with the minimal arrival time to the destination is meant
                # to reduce the total nodes being added to the nodes_2_update list

                min_dist = np.inf
                min_link = -1
                for link, active in enumerate(links_2_update):
                    if active:
                        if arrival_maps[destination, t, link] <= min_dist:
                            min_link = link
                            min_dist = arrival_maps[destination, t, link]
                links_2_update[min_link] = False  # no longer considered
                # _log('deactivated node ' + str(min_node))
                new_dist = np.inf
                for out_link, turn in zip(out_turns.get_row(min_link), out_turns.get_nnz(min_link)):
                    if t + np.uint32(turn_time[t, turn]) >= tot_time_steps - 1:
                        dist = arrival_maps[destination, tot_time_steps - 1, out_link] + new_costs[t, turn] - \
                               (tot_time_steps - 1 - t) * step_size
                    else:
                        dist = (1 - interpolation_frac[t, turn]) * arrival_maps[
                            destination, t + np.uint32(turn_time[t, turn]), out_link] + interpolation_frac[
                                   t, turn] * arrival_maps[
                                   destination, t + np.uint32(turn_time[t, turn]) + 1, out_link]
                    # _log(f'distance to {min_node} via out_link node {to_node[link]} is {dist} ')
                    if dist < new_dist:
                        new_dist = dist
                        assert new_dist>0
                        assert new_dist<1000

                # _log(f'result for node {min_node} written back? {np.abs(new_dist - arrival_maps[destination, t, min_node]) > route_choice_delta}')
                if np.abs(new_dist - arrival_maps[destination, t, min_link]) > route_choice_delta:
                    # new arrival time found
                    arrival_maps[destination, t, min_link] = new_dist
                    for turn in in_turns.get_nnz(min_link):
                        links_2_update[from_link[turn]] = True
                        next_links_to_update[from_link[turn]] = True


# TODO: test the @njit(parallel=True) option here
@njit(cache=True)
def get_turning_fractions(dynamic_demand: InternalDynamicDemand, network: Network, time: SimulationTime, arrival_maps,
                          new_costs, departure_time_offset=route_choice_agg):
    """
    Calculates turning fractions taking into account closed turns. Deterministic procedure; a turn has a fraction of 1
    if it is on the destination based shortest path tree, zero otherwise.
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
    turning_fractions = np.zeros((dynamic_demand.tot_active_destinations, time.tot_time_steps, network.tot_turns),
                                 dtype=np.float64)
    for dest_idx in range(dynamic_demand.all_active_destinations.size):
        # print(f'destination {dynamic_demand.all_active_destinations[dest_idx]}')
        for t in range(time.tot_time_steps):
            for link in range(network.tot_links):
                min_dist = np.inf
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
                    if dist <= min_dist:
                        min_turn = out_turn
                        min_dist = dist
                if min_turn != -1:
                    turning_fractions[dest_idx, t, min_turn] = 1
    return turning_fractions


@njit(parallel=True, cache=True)
def link_to_turn_costs(link_costs: np.ndarray, out_links: UI32CSRMatrix,
                       in_turns: UI32CSRMatrix, tot_turns, time: SimulationTime,
                       turn_delays,
                       use_turn_delays=use_turn_delays):
    #  the turn costs are defined as the cost incurred on the from link + the turn delay
    # it does NOT include the travel time on the to_link of the turn
    # the turn delay itself is defined as the time it takes to make the turn on the node
    # this additional delay is not yet taken account in the propagation and by default 0 for all turns
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
    if use_turn_delays:
        for t in range(time.tot_time_steps):
            for node in prange(out_links.get_nnz_rows().size):
                # turn and link labelling follows the node labelling
                # turns with the same via node are labelled consecutively
                # the same is usually true for the outgoing links of a node (if it's not a connector)
                for to_link in out_links.get_nnz(node):
                    for turn, from_link in zip(in_turns.get_nnz(to_link), in_turns.get_row(to_link)):
                        turn_costs[t, turn] += link_costs[t, from_link]
                        interpolation_fraction = link_costs[t, from_link] / time.step_size
                        if t + 2 + np.floor(interpolation_fraction) > time.tot_time_steps:
                            turn_costs[t, turn] += turn_delays[-1, turn]
                        elif interpolation_fraction < 1:
                            turn_costs[t, turn] += np.interp(interpolation_fraction, [0, 1], turn_delays[t:t + 2, turn])
                        else:
                            arrival_period = np.int32(np.floor(interpolation_fraction))
                            interpolation_fraction = interpolation_fraction - arrival_period
                            turn_costs[t, turn] += np.interp(interpolation_fraction, [0, 1],
                                                             turn_delays[arrival_period:arrival_period + 2, turn])
    else:
        # no interpolation needed
        for node in prange(out_links.get_nnz_rows().size):
            for to_link in out_links.get_nnz(node):
                for turn, from_link in zip(in_turns.get_nnz(to_link), in_turns.get_row(to_link)):
                    turn_costs[:, turn] = link_costs[:, from_link]

    return turn_costs


@njit(cache=True)
def link_to_turn_costs_deterministic(link_costs: np.ndarray, out_links: UI32CSRMatrix,
                                     in_turns: UI32CSRMatrix, tot_turns, time: SimulationTime, link_types,
                                     turning_fractions, ff_tt, cvn_up, turn_restrictions):
    tot_time_steps = link_costs.shape[0]
    turn_costs = np.zeros((tot_time_steps, tot_turns), dtype=np.float32)
    for t in range(time.tot_time_steps):
        for node in prange(out_links.get_nnz_rows().size):
            for to_link in out_links.get_nnz(node):
                for turn, from_link in zip(in_turns.get_nnz(to_link), in_turns.get_row(to_link)):
                    if turn_restrictions[turn]:
                        turn_costs[t, turn] = restricted_turn_cost
                    elif link_types[from_link] != 1:
                        turn_costs[t, turn] = link_costs[t, from_link]
                    else:
                        # currently not used ..
                        congestion_cost = link_costs[t, from_link] - ff_tt[from_link]
                        if np.sum(turning_fractions[:, t, turn] * cvn_up[t, from_link, :]) > 0:
                            turn_costs[t, turn] = congestion_cost + ff_tt[from_link]
                        else:
                            turn_costs[t, turn] = congestion_cost + ff_tt[from_link]
    return turn_costs
