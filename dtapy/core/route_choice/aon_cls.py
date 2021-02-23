#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import float32, uint32
from numba.experimental import jitclass

spec_aon_state = [('cur_costs', float32[:, :]),
                  ('prev_costs', float32[:, :]),
                  ('arrival_maps', float32[:, :, :]),
                  ('turning_fractions', float32[:,:,:]),
                  ('interpolation_frac', float32[:,:]),
                  ('link_time', float32[:,:]),
                  ('parents'), uint32[:,:]]


@jitclass(spec_aon_state)
class AONState(object):
    def __init__(self, cur_costs, prev_costs, arrival_maps, turning_fractions, interpolation_fraction, link_time,parents ):
        """
        Parameters
        ----------
        cur_costs : float32 array, time_steps x links
        prev_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        parents : uint32 array, destination x time_steps x nodes
        """
        self.prev_costs = prev_costs
        self.cur_costs = cur_costs
        self.arrival_maps = arrival_maps
        self.parents =  parents
        self.turning_fractions =  turning_fractions
        self.interpolation_frac = interpolation_fraction
        self.link_time= link_time

