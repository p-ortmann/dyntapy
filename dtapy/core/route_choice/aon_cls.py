#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import float32, uint32
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from dtapy.datastructures.csr import f32csr_type

spec_aon_state = [('cur_costs', float32[:, :]),
                  ('prev_costs', float32[:, :]),
                  ('arrival_maps', float32[:, :, :]),
                  ('turning_fractions', float32[:,:,:]),
                  ('interpolation_frac', float32[:,:]),
                  ('link_time', float32[:,:]),
                  ('connector_choice',ListType(f32csr_type))]


@jitclass(spec_aon_state)
class AONState(object):
    def __init__(self, cur_costs, prev_costs, arrival_maps, turning_fractions, interpolation_fraction, link_time, connector_choice ):
        """
        Parameters
        ----------
        cur_costs : float32 array, time_steps x links
        prev_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        connector_choice : F32CSRMatrix, connector x destinations
        """
        self.prev_costs = prev_costs
        self.cur_costs = cur_costs
        self.arrival_maps = arrival_maps
        self.turning_fractions =  turning_fractions
        self.interpolation_frac = interpolation_fraction
        self.link_time= link_time
        self.connector_choice = connector_choice

