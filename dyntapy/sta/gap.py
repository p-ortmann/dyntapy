#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from numba import njit

from dyntapy.settings import parameters

gap_method = parameters.static_assignment.gap_method


@njit
def gap(flows, travel_times, demand, sp_costs, method=gap_method):
    """
    gap functions typically used in DUE assignments

            Parameters
            ----------
            method : string, specifies which gap function to use
            flows : link flow vector
            travel_times : link travel time vector given the flow
            demand : vector of OD demands
            sp_costs : vector of shortest path costs
            between OD given the current network load
            -------
    """
    gap_methods = ["relative", "avg_excess_cost"]
    assert method in gap_methods
    if method == "relative":
        gamma = __relative(flows, travel_times, demand, sp_costs) - 1
    elif method == "avg_excess_cost":
        gamma = __avg_excess_cost(flows, travel_times, demand, sp_costs)

    return gamma


@njit
def __relative(
    flows: np.ndarray,
    travel_times: np.ndarray,
    demand: np.ndarray,
    sp_costs: np.ndarray,
):
    return np.vdot(flows.astype(np.float32), travel_times.astype(np.float32)) / np.vdot(
        demand.astype(np.float32), sp_costs.astype(np.float32)
    )


@njit
def __avg_excess_cost(flows, travel_times, demand, sp_costs):
    return (
        np.vdot(flows.astype(np.float32), travel_times.astype(np.float32))
        - np.vdot(demand.astype(np.float32), sp_costs.astype(np.float32))
    ) / np.sum(demand)
