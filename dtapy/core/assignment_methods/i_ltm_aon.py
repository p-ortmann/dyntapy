#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from dtapy.core.route_choice.aon_setup import setup_aon
from dtapy.core.route_choice.aon import calc_turning_fractions, calc_source_connector_choice
from dtapy.core.network_loading.link_models.i_ltm_setup import i_ltm_setup
from dtapy.core.network_loading.link_models.i_ltm import i_ltm
from dtapy.core.supply import Network
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime

def i_ltm_aon(network: Network, dynamic_demand: InternalDynamicDemand, route_choice_time: SimulationTime,
              network_loading_time: SimulationTime):
    aon_state = setup_aon(network, route_choice_time, dynamic_demand)
    # aon_state is updated in this routine
    print('aon passed')

    calc_turning_fractions(dynamic_demand, network, route_choice_time, aon_state)
    print('calc turnf')
    calc_source_connector_choice(network, aon_state, dynamic_demand)
    print('calc c choice')
    iltm_state, network = i_ltm_setup(network, network_loading_time, dynamic_demand)
    i_ltm(network, dynamic_demand, iltm_state, network_loading_time, aon_state.turning_fractions, aon_state.connector_choice)
    print(' iltm passing yahy')
    print('hi')