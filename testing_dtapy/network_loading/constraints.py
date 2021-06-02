#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from dtapy.settings import parameters
from dtapy.datastructures.csr import UI32CSRMatrix
import numpy as np
from numpy import ndarray
from numba import njit, prange

precision = parameters.network_loading.precision


#@njit(cache=True)
def continuity(cvn_up: ndarray, cvn_down: ndarray, in_links: UI32CSRMatrix,
               out_links: UI32CSRMatrix, max_delta: float = precision):
    """

    Parameters
    ----------
    cvn_up : upstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    cvn_down : downstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    in_links : CSR node x links
    out_links : CSR node x links
    max_delta : float, allowed constraint violation

    Returns
    -------
    """
    tot_time_steps = cvn_down.shape[0]
    tot_destinations = cvn_down.shape[2]
    for t in prange(tot_time_steps):
        for d in prange(tot_destinations):
            for node in range(in_links.nnz_rows.size):
                in_flow = 0.0
                out_flow = 0.0
                for in_link in in_links.get_nnz(node):
                    in_flow += cvn_down[t, in_link, d]
                for out_link in out_links.get_nnz(node):
                    out_flow += cvn_up[t, out_link, d]
                if np.abs(out_flow - in_flow) > max_delta:
                    print("continuity violation in node " + str(node) +
                          " at time " + str(t) + " for destination id " + str(d))
                    raise ValueError()
    print('continuity test passed successfully')
