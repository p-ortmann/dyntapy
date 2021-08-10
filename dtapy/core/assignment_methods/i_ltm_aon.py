#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

import numpy as np

from dtapy.core.demand import InternalDynamicDemand, Demand
from dtapy.core.network_loading.link_models.i_ltm import i_ltm
from dtapy.core.network_loading.link_models.i_ltm_setup import i_ltm_aon_setup
from dtapy.core.network_loading.link_models.utilities import cvn_to_flows, _debug_plot, cvn_to_travel_times
from dtapy.core.route_choice.deterministic import update_route_choice
from dtapy.core.route_choice.aon_setup import setup_aon
from dtapy.core.route_choice.aon import link_to_turn_costs_deterministic
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from dtapy.settings import parameters
from dtapy.utilities import _log
from dtapy.core.route_choice.aon import update_arrival_maps
from dtapy.core.debugging import sum_of_turning_fractions, verify_assignment_state
from dtapy.datastructures.csr import UI32CSRMatrix
from numba.typed import List
from numba import njit
from dtapy.settings import debugging

smooth_turning_fractions = parameters.assignment.smooth_turning_fractions
smooth_costs = parameters.assignment.smooth_costs
max_iterations = parameters.assignment.max_iterations


# @njit(cache=True)
def i_ltm_aon(network: Network, dynamic_demand: InternalDynamicDemand, route_choice_time: SimulationTime,
              network_loading_time: SimulationTime):
    convergence = List()
    convergence.append(np.inf)
    _log('setting up data structures for i_ltm', to_console=True)
    iltm_state, network = i_ltm_aon_setup(network, network_loading_time, dynamic_demand)
    _log('initializing AON', to_console=True)
    aon_state = setup_aon(network, route_choice_time, dynamic_demand)
    k = 1
    converged = False
    sum_of_turning_fractions(aon_state.turning_fractions, network.links.out_turns, network.links.link_type,
                             network.turns.to_node, tot_centroids=dynamic_demand.tot_centroids)
    turn_delays = np.full((route_choice_time.tot_time_steps,network.tot_turns),network.turns.t0)
    while k < 1001 and not converged:
        _log('calculating network state in iteration ' + str(k), to_console=True)
        i_ltm(network, dynamic_demand, iltm_state, network_loading_time, aon_state.turning_fractions, k)
        verify_assignment_state(network, aon_state.turning_fractions, iltm_state.cvn_up, iltm_state.cvn_down,
                                dynamic_demand.tot_centroids)
        link_costs = cvn_to_travel_times(cvn_up=np.sum(iltm_state.cvn_up, axis=2),
                                         cvn_down=np.sum(iltm_state.cvn_down, axis=2),
                                         time=network_loading_time,
                                         network=network, con_down=iltm_state.con_down)
        turn_costs = link_to_turn_costs_deterministic(link_costs, network.nodes.out_links, network.links.in_turns, network.tot_turns,
                                        route_choice_time,network.links.link_type,aon_state.turning_fractions,
                                        network.links.length/network.links.v0,iltm_state.cvn_up)
        _log('updating arrival in iteration ' + str(k), to_console=True)
        update_arrival_maps(network, network_loading_time, dynamic_demand, aon_state.arrival_maps, aon_state.turn_costs,
                            turn_costs)
        if debugging:
            _rc_debug_plot(iltm_state, network, network_loading_time, aon_state, link_costs,
                           title=f'RC state in iteration {k}')
        _log('updating route choice in iteration ' + str(k), to_console=True)
        gec = update_route_choice(aon_state, turn_costs, iltm_state.cvn_down,  network, dynamic_demand, route_choice_time, k)
        if k > 1:
            convergence.append(np.sum(gec))
            _log('new flows, gap is  : ' + str(gec), to_console=True)
            if np.all(gec<0.001):
                converged=True
        k = k + 1
        sum_of_turning_fractions(aon_state.turning_fractions, network.links.out_turns, network.links.link_type,
                                 network.turns.to_node, tot_centroids=dynamic_demand.tot_centroids)
    flows = cvn_to_flows(iltm_state.cvn_down)
    convergence_arr = np.empty(len(convergence))
    for _id, i in enumerate(convergence):
        convergence_arr[_id] = i
    return flows, link_costs



# @njit(cache=True)
def is_cost_converged(costs, flows, arrival_map, dynamic_demand: InternalDynamicDemand, step_size,
                      out_links: UI32CSRMatrix,
                      target_gap=parameters.assignment.gap):
    """

    Parameters
    ----------
    out_links :
    step_size : np.float32, duration of a time step
    target_gap : np.float64, threshold for convergence
    costs : tot_time_steps x tot_links
    flows : tot_time_steps x tot_links
    arrival_map : tot_destinations x tot_time_steps x tot_nodes
    dynamic_demand : InternalDynamicDemand

    Returns
    -------
    Tuple(boolean, np.float64)

    """
    experienced_travel_times = np.sum(np.multiply(costs.astype(np.float64), flows.astype(np.float64)))
    shortest_path_travel_times = np.float64(0)
    for t in dynamic_demand.loading_time_steps:
        demand: Demand = dynamic_demand.get_demand(t)
        for origin in demand.to_destinations.get_nnz_rows():
            for flow, destination in zip(demand.to_destinations.get_row(origin),
                                         demand.to_destinations.get_nnz(origin)):
                shortest_path_travel_times += flow * (arrival_map[
                                                          np.flatnonzero(
                                                              dynamic_demand.all_active_destinations == destination)[
                                                              0], t + 1, out_links.get_nnz(origin)[0]] - (
                                                                  t + 1) * step_size)
    gap_value = np.divide(experienced_travel_times, shortest_path_travel_times) - 1
    return gap_value < target_gap, gap_value


def _rc_debug_plot(results, network, time, rc_state, link_costs, title='None', highlight_nodes=[], toy_network=True):
    from dtapy.visualization import show_dynamic_network
    from dtapy.__init__ import current_network
    flows = cvn_to_flows(results.cvn_down)
    toy_network = True
    cur_queues = np.sum(results.cvn_up, axis=2) - np.sum(results.cvn_down, axis=2)  # current queues
    show_dynamic_network(current_network, time, toy_network=toy_network, title=title, link_kwargs=
    {'cvn_up': results.cvn_up, 'cvn_down': results.cvn_down, 'vind': network.links.vf_index,
     'wind': network.links.vw_index, 'flows': flows, 'current_queues': cur_queues, 'costs': link_costs},
                         highlight_nodes=highlight_nodes)
