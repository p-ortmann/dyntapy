#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import njit, prange

from dyntapy.dta.time import SimulationTime
from dyntapy.settings import parameters
from dyntapy.supply import Network

epsilon = parameters.dynamic_assignment.network_loading.epsilon


@njit(cache=True, parallel=True)
def cvn_to_travel_times(
    cvn_up: np.ndarray,
    cvn_down: np.ndarray,
    con_down: np.ndarray,
    time: SimulationTime,
    network: Network,
):
    """
    Calculates travel times per link based on
    single commodity cumulative vehicle numbers
    follows Long, Jiancheng, Ziyou Gao, and W.Y.Szeto.
    “Discretised Link Travel Time Models Based on Cumulative
    Flows: Formulations and Properties.”
    Transportation Research Part B: Methodological
    45, no. 1(January 1, 2011): 232–54.
    https: // doi.org / 10.1016 / j.trb.2010.05.002.
    See DUE issues ticket #17 in gitlab for design decisions taken here
    Parameters
    ----------
    cvn_up :2D array, tot_time_steps x tot_links
    cvn_down : 2D array, tot_time_steps x tot_links
    time : SimulationTime
    network : Network

    Returns
    -------
    travel_times: 2D array, tot_time_steps x tot_links
    """
    travel_times = np.zeros((time.tot_time_steps, network.tot_links), dtype=np.float32)
    for link in prange(network.tot_links):
        ff_tt = np.float32(
            (network.links.length[link] / network.links.free_speed[link])
        )
        travel_times[:, link] = ff_tt
        if network.links.link_type[link] == -1:
            # costs on sink connectors are always free flow
            continue
        else:
            previous_interval_vehicle_travel_time = (
                ff_tt  # keeps track of last travel time across timesteps
            )
            for t in range(time.tot_time_steps):
                if not con_down[t, link]:
                    continue
                else:
                    last_vehicle_to_enter = cvn_up[t, link]
                    if t == 0:
                        first_vehicle_to_enter = 0
                    else:
                        first_vehicle_to_enter = cvn_up[t - 1, link]
                    delta_cvn = last_vehicle_to_enter - first_vehicle_to_enter
                    last_exit_time = find_exit_time(
                        last_vehicle_to_enter, cvn_down[:, link], t, time.tot_time_steps
                    )
                    if np.uint32(last_exit_time) < last_exit_time:
                        after_last_exit_interval = np.uint32(last_exit_time) + 1
                    else:
                        after_last_exit_interval = np.uint32(last_exit_time)
                    travel_time_average = 0.0
                    first_vehicle_to_enter_in_k = first_vehicle_to_enter
                    if not con_down[t - 1, link]:
                        previous_interval_vehicle_travel_time = ff_tt
                    for k in range(t, after_last_exit_interval):
                        # k is the running index for intervals
                        # in which vehicles from k left the link
                        if cvn_down[k, link] > first_vehicle_to_enter_in_k:
                            if cvn_down[k, link] > last_vehicle_to_enter:
                                vehicles_in_flow_packet = (
                                    last_vehicle_to_enter - first_vehicle_to_enter_in_k
                                )
                                vehicle_entry = t + 1
                                if not con_down[k, link]:
                                    # if there is no congestion downstream travel times
                                    # get overestimated dramatically due to
                                    # interpolation, we apply capacity discharge
                                    current_interval_vehicle_travel_time = max(
                                        (
                                            k
                                            + vehicles_in_flow_packet
                                            / network.links.capacity[link]
                                            - vehicle_entry
                                        )
                                        * time.step_size,
                                        ff_tt,
                                    )
                                else:
                                    current_interval_vehicle_travel_time = max(
                                        (
                                            k
                                            + vehicles_in_flow_packet
                                            / (
                                                cvn_down[k, link]
                                                - cvn_down[k - 1, link]
                                            )
                                            - vehicle_entry
                                        )
                                        * time.step_size,
                                        ff_tt,
                                    )

                            else:
                                vehicles_in_flow_packet = (
                                    cvn_down[k, link] - first_vehicle_to_enter_in_k
                                )
                                vehicle_entry = find_entry_time(
                                    cvn_down[k, link], cvn_up[:, link], k
                                )
                                current_interval_vehicle_travel_time = max(
                                    (k + 1 - vehicle_entry) * time.step_size, ff_tt
                                )
                            travel_time_average += (
                                vehicles_in_flow_packet
                                / delta_cvn
                                * (
                                    current_interval_vehicle_travel_time
                                    + previous_interval_vehicle_travel_time
                                )
                                / 2
                            )
                            first_vehicle_to_enter_in_k = cvn_down[
                                k, link
                            ]  # first vehicle of the next period
                            previous_interval_vehicle_travel_time = (
                                current_interval_vehicle_travel_time
                            )
                            # is the last of the previous period.
                    travel_times[t, link] = travel_time_average
    return travel_times


@njit(cache=True)
def find_entry_time(cvn: float, cvn_up: np.ndarray, k: int):
    # find the time of entry of the cumulative cvn that left the link during
    # interval k, e.g. in [k,k+1. returns entry time in interval units.
    interval_before_entry = -1
    for t in range(k, -1, -1):
        if cvn_up[t] < cvn:
            interval_before_entry = t
            break
    # if the condition was triggered during none of the intervals
    # the vehicle must've entered during the
    # very first interval
    if interval_before_entry == -1:
        entry_time = cvn / cvn_up[0]
    else:
        entry_time = (
            (cvn - cvn_up[interval_before_entry])
            / (cvn_up[interval_before_entry + 1] - cvn_up[interval_before_entry])
            + interval_before_entry
            + 1
        )
    return entry_time


@njit(cache=True)
def find_exit_time(cvn: np.float32, cvn_down: np.ndarray, k: int, T: int):
    # find the time of exit of the cumulative cvn that
    # entered the link during interval k, e.g. in [k,k+1]
    # returns exit time in interval units
    exit_interval = T
    for t in range(k, T):  # range ends at T-1
        if cvn_down[t] >= cvn:
            exit_interval = t
            break
    # if the condition was triggered during none of the
    # intervals the vehicle must've exited after the
    # simulation ended
    exit_interval = np.uint32(exit_interval)
    if exit_interval == T:
        return np.float32(T)  # needs to be float, consistent return type with below
    else:
        exit_time = exit_interval + (cvn - cvn_down[exit_interval - 1]) / (
            cvn_down[exit_interval] - cvn_down[exit_interval - 1]
        )
        return exit_time
