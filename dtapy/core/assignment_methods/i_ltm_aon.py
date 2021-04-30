#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

import numpy as np

from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.network_loading.link_models.i_ltm import i_ltm
from dtapy.core.network_loading.link_models.i_ltm_setup import i_ltm_setup
from dtapy.core.network_loading.link_models.utilities import cvn_to_flows, _debug_plot, cvn_to_travel_times
from dtapy.core.route_choice.aon_cls import update_route_choice
from dtapy.core.route_choice.aon_setup import setup_aon
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from dtapy.settings import parameters
from dtapy.utilities import _log
from dtapy.core.route_choice.aon import update_arrival_maps

smooth_turning_fractions = parameters.assignment.smooth_turning_fractions
smooth_costs = parameters.assignment.smooth_costs
max_iterations = parameters.assignment.max_iterations


# @njit(cache=True)
def i_ltm_aon(network: Network, dynamic_demand: InternalDynamicDemand, route_choice_time: SimulationTime,
              network_loading_time: SimulationTime):
    _log('initializing AON', to_console=True)
    aon_state = setup_aon(network, route_choice_time, dynamic_demand)
    _log('setting up data structures for i_ltm', to_console=True)
    iltm_state, network = i_ltm_setup(network, network_loading_time, dynamic_demand)
    k = 1
    converged = False
    old_flows = np.zeros((network_loading_time.tot_time_steps, network.tot_links), dtype=np.float32)
    while k < 1001 and not converged:
        _log('calculating network state in iteration ' + str(k), to_console=True)
        i_ltm(network, dynamic_demand, iltm_state, network_loading_time, aon_state.turning_fractions,
              aon_state.connector_choice, k)
        costs = cvn_to_travel_times(cvn_up=np.sum(iltm_state.cvn_up, axis=2),
                                    cvn_down=np.sum(iltm_state.cvn_down, axis=2),
                                    time=network_loading_time,
                                    network=network)
        new_flows = cvn_to_flows(iltm_state.cvn_down)
        if k > 1:
            converged = is_converged(old_flows, new_flows)
            _log('new flows, converged : ' + str(converged), to_console=True)
        old_flows = new_flows
        _log('updating route choice in iteration ' + str(k), to_console=True)
        k = k + 1
        update_arrival_maps(network, network_loading_time, dynamic_demand, aon_state.arrival_maps, aon_state.costs, costs)
        update_route_choice(aon_state, costs, network, dynamic_demand, route_choice_time, k, 'msa')
        if k==300:
            _rc_debug_plot(iltm_state, network, network_loading_time, aon_state, costs,
                           title=f'RC state in iteration {k}')
    print('finished it ' + str(k))
    _rc_debug_plot(iltm_state, network, network_loading_time, aon_state, costs,
                   title=f'RC state in iteration {k}')


    flows = cvn_to_flows(iltm_state.cvn_down)
    # costs = cvn_to_travel_times(cvn_up=np.sum(iltm_state.cvn_up, axis=2),
    #                             cvn_down=np.sum(iltm_state.cvn_down, axis=2),
    #                             time=network_loading_time,
    #                             network=network)
    return flows, costs


# @njit(cache=True)
def is_converged(old_flows: np.ndarray, new_flows: np.ndarray, gap=parameters.assignment.gap, method="relative_max"):
    """

    Parameters
    ----------
    gap : float
    old_flows : tot_time_steps x tot_links
    new_flows : tot_time_steps x tot_links
    method : str
    Returns
    -------
    returns True if flows are converged under the given criteria, False otherwise
    """
    if method == 'relative_max':
        result_gap = np.abs(new_flows - old_flows) / old_flows
        violated = result_gap > gap
        # current_gap = np.nanmax(gap)
        either_non_zero = np.logical_or(old_flows > 0, new_flows > 0)
        return not np.any(np.logical_and(violated, either_non_zero))
    else:
        raise NotImplementedError


def _rc_debug_plot(results, network, time, rc_state, link_costs, title='None', toy_network=True):
    from dtapy.visualization import show_assignment
    from dtapy.__init__ import current_network
    flows = cvn_to_flows(results.cvn_down)
    cur_queues = np.sum(results.cvn_up, axis=2) - np.sum(results.cvn_down, axis=2)  # current queues
    show_assignment(current_network, time, toy_network=toy_network, title=title, link_kwargs=
    {'cvn_up': results.cvn_up, 'cvn_down': results.cvn_down, 'vind': network.links.vf_index,
     'wind': network.links.vw_index, 'flows': flows, 'current_queues': cur_queues, 'costs': link_costs},
                    node_kwargs={'arrival': rc_state.arrival_maps.transpose(1, 2, 0)})
