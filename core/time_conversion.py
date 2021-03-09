#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# utilities that allow us to go between route choice and network loading methods with different time discretization
# mainly a switch from link costs in network loading time to route choice time
# and the inverse for turning fractions
import numpy as np
import numba


# if this becomes more common, we may want to re-implement Scipys functions here..
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

# some functions that switch outputs from route choice time discretization to network loading time discretization and
# vice versa
# not tested yet
# THIS IS JUST A PROTOTYPE

# TODO: add conversion for connector choice representation
@numba.njit()
def __turning_fractions(turning_fractions, T, step_function=True):
    """

    Parameters
    ----------
    turning_fractions : dim destination, tot_time_steps, tot_turns
    T : new tot_time_steps to discretize to
    step_function: boolean, whether to yield turning fractions

    Returns
    -------
    """
    new_turning_fractions = np.empty_like(turning_fractions)
    T0 = turning_fractions.shape[1]
    dt = np.div(T0 / T)
    new_samples = np.arange(T) * dt
    if not step_function:
        for t in new_samples:
            t0 = np.floor(t)
            interpolation_frac = t - t0
            new_turning_fractions[t] = (1 - interpolation_frac) * turning_fractions[:, t0,
                                                                  :] + interpolation_frac * turning_fractions[:, t0 + 1,
                                                                                            :]
    else:
        for t in new_samples:
            t0 = np.round(t)  # simply maps to nearest point in previous discretization
            new_turning_fractions[t] = turning_fractions[:, t0, :]


@numba.njit()
def __link_costs(link_costs, T, step_function=True):
    """

    Parameters
    ----------
    link_costs : dim  tot_time_steps, tot_links
    T : new tot_time_steps to discretize to
    step_function: boolean, whether to yield turning fractions

    Returns
    -------
    """
    new_link_costs = np.empty_like(link_costs)
    T0 = link_costs.shape[1]
    dt = np.div(T0 / T)
    new_samples = np.arange(T) * dt
    if not step_function:
        for t in new_samples:
            t0 = np.floor(t)
            interpolation_frac = t - t0
            new_link_costs[t] = (1 - interpolation_frac) * link_costs[:, t0,
                                                           :] + interpolation_frac * link_costs[:, t0 + 1,
                                                                                     :]
    else:
        for t in new_samples:
            t0 = np.round(t)  # simply maps to nearest point in previous discretization
            new_link_costs[t] = link_costs[:, t0, :]
