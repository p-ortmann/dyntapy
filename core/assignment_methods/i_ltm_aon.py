#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from core.route_choice.aon_setup import setup_aon
from core.route_choice.aon import calc_turning_fractions
from core.network_loading.link_models.i_ltm_setup import i_ltm_setup
from core.assignment_cls import Network, InternalDynamicDemand, SimulationTime

def i_ltm_aon(network: Network, dynamic_demand: InternalDynamicDemand, route_choice_time: SimulationTime, network_loading_time:SimulationTime):


    aon_state = setup_aon(network,route_choice_time, dynamic_demand)
    # aon_state is updated in this routine
    print('aon passed')
    calc_turning_fractions(dynamic_demand, network, route_choice_time, aon_state)
    print('calc turnf')

