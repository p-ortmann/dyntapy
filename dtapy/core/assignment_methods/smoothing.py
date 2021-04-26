#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from numba import njit
import numpy as np


def smooth_arrays(current: np.ndarray, previous: np.ndarray, k, method='msa'):
    """

    Parameters
    ----------
    current : array from the current iteration
    previous : array of previous iteration
    k: int, iteration counter
    method : which smoothing method to use

    Returns
    -------

    """
    if current.shape != previous.shape:
        raise ValueError('cannot smooth on arrays with inconsistent dimensions')
    if method == 'msa':
        factor = 1 / np.power(k, (1 / 4))
        dx = current - previous
        return previous + factor * dx
    else:
        raise NotImplementedError


def smooth_sparse(current, previous, k, method='msa'):
    """
    smooths between two CSR matrices, updates previous with the result of the smoothing and returns it.
    ----------
    current : CSRMatrix, as defined in dtapy.datastructures
    previous : CSRMatrix, as defined in dtapy.datastructures
    k : int, iteration counter
    method : str, {'msa'}

    Returns
    -------
    """
    if method == 'msa':
        factor = 1 / np.power(k, (1 / 4))
        dx = current.values - previous.values
        previous.values = previous.values + factor * dx
        return previous
    else:
        raise NotImplementedError
