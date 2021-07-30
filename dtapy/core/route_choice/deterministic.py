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
from dtapy.core.assignment_methods.smoothing import smooth_sparse, smooth_arrays
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.route_choice.aon import get_turning_fractions, update_arrival_maps
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from dtapy.datastructures.csr import f32csr_type
from dtapy.settings import parameters
from dtapy.core.route_choice.qr_projection import qr_projection

smoothing_method = parameters.assignment.smooth_costs
spec_rc_state = [('link_costs', float32[:, :]),
                 ('turn_costs', float32[:,:]),
                 ('arrival_maps', float32[:, :, :]),
                 ('turning_fractions', float32[:, :, :]),
                 ('connector_choice', ListType(f32csr_type))]


@jitclass(spec_rc_state)
class RouteChoiceState(object):
    def __init__(self, link_costs, turn_costs, arrival_maps, turning_fractions):
        """
        Parameters
        ----------
        link_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        """
        self.link_costs = link_costs
        self.turn_costs= turn_costs
        self.arrival_maps = arrival_maps
        self.turning_fractions = turning_fractions

# @njit
def update_route_choice(state, turn_costs: np.ndarray, network: Network, dynamic_demand: InternalDynamicDemand,
                        time: SimulationTime, k: int, method='quasi-reduced-projection'):
    """

    Parameters
    ----------
    state : RouteChoiceState
    turn_costs : time_steps x links
    network : Network
    dynamic_demand : InternalDynamicDemand
    time : SimulationTime
    k : int, number of iteration
    method : str


    """
    if method == 'msa':
        # deterministic case non convergent, saw tooth pattern settles in ..
        update_arrival_maps(network, time, dynamic_demand, state.arrival_maps, state.turn_costs, turn_costs)
        turning_fractions = get_turning_fractions(dynamic_demand, network, time, state.arrival_maps, turn_costs)
        state.turning_fractions = smooth_arrays(turning_fractions, state.turning_fractions, k, method)
        state.turn_costs = turn_costs
    if method == 'quasi-reduced-projection':
        # deterministic approach of updating the turning fractions, see willem's thesis chapter 4 for background
        # should lead to smooth convergence
        qr_projection()


    else:
        raise NotImplementedError
