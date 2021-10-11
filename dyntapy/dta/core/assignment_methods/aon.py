#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

import numpy as np
from dyntapy.dta.core.demand import InternalDynamicDemand
from dyntapy.dta.core.network_loading.link_models.i_ltm import i_ltm
from dyntapy.dta.core.network_loading.link_models.i_ltm_setup import i_ltm_aon_setup
from dyntapy.dta.core.network_loading.link_models.utilities import cvn_to_flows, cvn_to_travel_times
from dyntapy.dta.core.route_choice.aon_setup import get_aon_route_choice
from dyntapy.dta.core.supply import Network
from dyntapy.dta.core.time import SimulationTime
from numba import njit


@njit(cache=True)
def aon(network: Network, dynamic_demand: InternalDynamicDemand, time: SimulationTime):
    iltm_state, network = i_ltm_aon_setup(network, time, dynamic_demand)
    aon_state = get_aon_route_choice(network, time, dynamic_demand)
    i_ltm(network, dynamic_demand, iltm_state, time, aon_state.turning_fractions)
    link_costs = cvn_to_travel_times(cvn_up=np.sum(iltm_state.cvn_up, axis=2),
                                     cvn_down=np.sum(iltm_state.cvn_down, axis=2),
                                     time=time,
                                     network=network, con_down=iltm_state.con_down)

    flows = cvn_to_flows(iltm_state.cvn_down)
    return flows, link_costs
