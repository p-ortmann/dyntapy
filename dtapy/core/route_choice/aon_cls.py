#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import float32
from numba.experimental import jitclass

spec_aon_state = [('cur_costs', float32[:, :]),
                  ('prev_costs', float32[:, :]),
                  'arrival_maps', float32[:, :, :]]


@jitclass(spec_aon_state)
class AONState(object):
    def __init__(self, cur_costs, prev_costs, arrival_maps):
        """
        Parameters
        ----------
        cur_costs : float32 array, links x time_steps
        prev_costs : float32 array, links x time_steps
        arrival_maps : float32 array, nodes x time_steps
        """
        self.prev_costs = prev_costs
        self.cur_costs = cur_costs
        self.arrival_maps = arrival_maps