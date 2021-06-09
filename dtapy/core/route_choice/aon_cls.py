#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import float32
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from numba.typed import List

from dtapy.core.assignment_methods.smoothing import smooth_sparse, smooth_arrays
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.route_choice.aon import get_turning_fractions, update_arrival_maps
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from dtapy.datastructures.csr import F32CSRMatrix
from dtapy.datastructures.csr import f32csr_type
from dtapy.settings import parameters

smoothing_method = parameters.assignment.smooth_costs
spec_rc_state = [('costs', float32[:, :]),
                 ('arrival_maps', float32[:, :, :]),
                 ('turning_fractions', float32[:, :, :]),
                 ('connector_choice', ListType(f32csr_type))]


@jitclass(spec_rc_state)
class RouteChoiceState(object):
    def __init__(self, cur_costs, arrival_maps, turning_fractions):
        """
        Parameters
        ----------
        cur_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        """
        self.costs = cur_costs
        self.arrival_maps = arrival_maps
        self.turning_fractions = turning_fractions

# @njit
def update_route_choice(state, costs: np.ndarray, network: Network, dynamic_demand: InternalDynamicDemand,
                        time: SimulationTime, k: int, method='msa'):
    """

    Parameters
    ----------
    state : RouteChoiceState
    costs : time_steps x links
    network : Network
    dynamic_demand : InternalDynamicDemand
    time : SimulationTime
    k : int, number of iteration
    method : str


    """
    print('hi from cost update')
    update_arrival_maps(network, time, dynamic_demand, state.arrival_maps, state.costs, costs)
    turning_fractions = get_turning_fractions(dynamic_demand, network, time, state.arrival_maps, costs)
    state.turning_fractions = smooth_arrays(turning_fractions, state.turning_fractions, k, method)
    state.costs = costs
