#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import float32, uint32
from numba.typed import List
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from dtapy.datastructures.csr import f32csr_type
from dtapy.core.route_choice.aon import get_source_connector_choice, get_turning_fractions, update_arrival_maps
from dtapy.core.assignment_methods.smoothing import smooth_sparse, smooth_arrays
from dtapy.datastructures.csr import F32CSRMatrix
from dtapy.settings import parameters

smoothing_method =  parameters.assignment.smooth_costs
spec_aon_state = [('costs', float32[:, :]),
                  ('arrival_maps', float32[:, :, :]),
                  ('turning_fractions', float32[:, :, :]),
                  ('connector_choice', ListType(f32csr_type))]


@jitclass(spec_aon_state)
class AONState(object):
    def __init__(self, cur_costs, arrival_maps, turning_fractions, connector_choice):
        """
        Parameters
        ----------
        cur_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        connector_choice : List<F32CSRMatrix>
        """
        self.costs = cur_costs
        self.arrival_maps = arrival_maps
        self.turning_fractions = turning_fractions
        self.connector_choice = connector_choice

    def update(self, costs, network, dynamic_demand, time, k, method='msa'):
        print('hi from cost update')
        update_arrival_maps(network, time, dynamic_demand, self.arrival_maps, self.costs, costs)
        turning_fractions = get_turning_fractions(dynamic_demand, network, time, self.arrival_maps, costs)
        connector_choice = List()
        for item in self.connector_choice:
            connector_choice.append(F32CSRMatrix(np.zeros_like(item.values), item.col_index, item.row_index))
        connector_choice = get_source_connector_choice(network, connector_choice,
                                                       self.arrival_maps, dynamic_demand)
        for t_id, (current, previous) in enumerate(zip(connector_choice, self.connector_choice)):
            connector_choice[t_id] = smooth_sparse(current, previous, k, method)
        self.connector_choice = connector_choice
        self.turning_fractions = smooth_arrays(turning_fractions, self.turning_fractions, k, method)

