#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from dtapy.core.route_choice.aon_setup import setup_aon
from dtapy.core.route_choice.aon import get_turning_fractions, get_source_connector_choice, update_arrival_maps
from dtapy.core.network_loading.link_models.i_ltm_setup import i_ltm_setup
from dtapy.core.network_loading.link_models.i_ltm import i_ltm
from dtapy.core.supply import Network
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime
from dtapy.utilities import _log, log
from dtapy.core.network_loading.link_models.utilities import cvn_to_flows, _debug_plot, cvn_to_travel_times
import numpy as np
from numba import njit
from dtapy.settings import parameters

smooth_turning_fractions = parameters.assignment.smooth_turning_fractions
smooth_costs = parameters.assignment.smooth_costs
max_iterations = parameters.assignment.max_iterations


# @njit(cache=True)
def i_ltm_aon(network: Network, dynamic_demand: InternalDynamicDemand, route_choice_time: SimulationTime,
              network_loading_time: SimulationTime):
    _log('initializing AON', to_console=True)
    aon_state = setup_aon(network, route_choice_time, dynamic_demand)
    iltm_state, network = i_ltm_setup(network, network_loading_time, dynamic_demand)

    iteration_counter = 0
    while iteration_counter < max_iterations:
        _log('calculating network state in iteration ' + str(iteration_counter), to_console=True)
        i_ltm(network, dynamic_demand, iltm_state, network_loading_time, aon_state.turning_fractions,
              aon_state.connector_choice)
        iteration_counter = max_iterations
        aon_state.costs = cvn_to_travel_times(cvn_up=np.sum(iltm_state.cvn_up, axis=2),
                                              cvn_down=np.sum(iltm_state.cvn_down, axis=2),
                                              time=network_loading_time,
                                              network=network)

        get_turning_fractions(dynamic_demand, network, route_choice_time, aon_state)
        _log('Calculating initial source connector choice', to_console=True)
        get_source_connector_choice(network, aon_state, dynamic_demand)

    flows = cvn_to_flows(iltm_state.cvn_down)

    _debug_plot(iltm_state, network, network_loading_time)

    return flows, new_costs
