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
from numba.experimental import jitclass

spec_time = [
    ("start", float32),
    ("end", float32),
    ("step_size", float32),
    ("tot_time_steps", uint32),
]


@jitclass(spec_time)
class SimulationTime(object):
    def __init__(self, start, end, step_size):
        """
        time discretization, units are always in hours
        Parameters
        ----------
        start : int
        end : int
        step_size : float
        """
        # TODO: start always 0
        self.start = start
        self.end = end
        self.step_size = step_size
        self.tot_time_steps = np.uint32(np.ceil((end - start) / step_size))
