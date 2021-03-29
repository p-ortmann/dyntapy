#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

import numpy as np


class Detectors():
    def __init__(self, x_coord, y_coord, speeds: np.ndarray, densities: np.ndarray = None, flows: np.ndarray = None):
        """

        Parameters
        ----------
        x_coord : longitude of all detectors
        y_coord : latitude of all detectors
        densities : Array, tot_detectors x tot_time_intervals
        speeds : Array, tot_detectors x tot_time_intervals
        flows : Array, tot_detectors x tot_time_intervals
        """
        self.tot_detectors = speeds.shape[0]
        self.tot_time_intervals = speeds.shape[1]
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.densities = densities
        self.speeds = speeds
        self.flows = flows
