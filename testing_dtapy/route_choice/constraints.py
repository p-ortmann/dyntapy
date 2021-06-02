#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from dtapy.datastructures.csr import UI32CSRMatrix
import numpy as np
from dtapy.settings import parameters
from numba import njit, prange
precision = parameters.route_choice.precision

def sum_of_turning_fractions(turning_fractions: np.ndarray, out_turns:UI32CSRMatrix, precision: float):
    """
    verifies if for each link the sum of the turning
    fractions for all outgoing turns is equal to 1.
    Parameters
    ----------
    turning_fractions : array, tot_active_destinations x tot_time_steps x tot_turns
    out_turns : CSR, link x link
    precision : float

    Returns
    -------

    """
    for t in prange(turning_fractions.shape[1]):
        for dest_id in prange(turning_fractions.shape[0]):
            for link in range(out_turns.nnz_rows.size):
                tf_sum=0.0
                for turn in out_turns.get_row(link):
                    tf_sum+= turning_fractions[dest_id,t,turn]
                if np.abs(tf_sum-1.0)>precision:
                    print("turning fraction sum violation for link " + str(link) +
                          " at time " + str(t) + " for destination id " + str(dest_id))
                    raise ValueError()


