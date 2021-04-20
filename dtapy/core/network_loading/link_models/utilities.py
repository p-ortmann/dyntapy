#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from dtapy.core.network_loading.link_models.i_ltm_cls import ILTMNetwork
from dtapy.visualization import show_assignment
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from numba import njit, prange, objmode
from dtapy.settings import parameters

epsilon = parameters.network_loading.epsilon


@njit(parallel=True)
def cvn_to_flows(cvn_down):
    """

    Parameters
    ----------
    cvn :

    Returns
    -------

    """
    tot_time_steps = cvn_down.shape[0]
    tot_links = cvn_down.shape[1]
    cvn_down = np.sum(cvn_down, axis=2)
    flows = np.zeros((tot_time_steps, tot_links), dtype=np.float32)
    flows[0, :] = cvn_down[0, :]
    for time in prange(1, tot_time_steps):
        flows[time, :] = np.abs(-cvn_down[time - 1, :] + cvn_down[time, :])
    return flows


# @njit(parallel=True)
def cvn_to_travel_times(cvn_up, cvn_down, time: SimulationTime, network: Network, method='forward'):
    """
    Calculates travel times per link based on single commodity cumulative vehicle numbers
    Parameters
    ----------
    cvn_up :2D array, tot_time_steps x tot_links
    cvn_down : 2D array, tot_time_steps x tot_links
    time : SimulationTime
    network : Network
    method : only 'forward' for now

    Returns
    -------
    travel_times: 2D array, tot_time_steps x tot_links
    If method == 'forward':
        travel_times[t,link] is the travel time experienced by the first vehicle
         that entered the link during the interval [t, t +dt]

    forward with the last vehicle does not work, due to i-ltm vehicle chopping with dissipating flows.
    #TODO: explore these differences..
    If method == 'backward_last':
        travel_times[t,link] is the travel time experienced by the last vehicle
         that left the link during the interval [t, t +dt]
    If method == 'backward_first':
        travel_times[t,link] is the travel time experienced by the first vehicle
         that left the link during the interval [t, t +dt]

    """
    travel_times = np.zeros((time.tot_time_steps, network.tot_links), dtype=np.float32)
    if method == 'forward':
        for t in prange(time.tot_time_steps):
            for link in prange(network.tot_links):
                if not network.links.link_type[link] == -1:
                    if np.sum(cvn_up[t, link] - cvn_up[t - 1, link]) > 1 \
                            or (t == 0 and cvn_up[0, link] > 1):
                        found_cvn = False
                        cvn = cvn_up[t - 1, link] + 1
                        if t == 0:
                            cvn = 1
                        for t2 in range(t, time.tot_time_steps, 1):
                            if cvn_down[t2, link] > cvn:
                                found_cvn = True
                                if t2 == 0:
                                    departure_time = 1 - \
                                                     (cvn_down[t2, link] - cvn) * 1 / (
                                                         cvn_down[t2, link])
                                else:
                                    departure_time = 1 - \
                                                     (cvn_down[t2, link] - cvn) * 1 / (
                                                             cvn_down[t2, link] -
                                                             cvn_down[
                                                                 t2 - 1, link])
                                    departure_time = (departure_time + t2 - 1)
                                travel_times[t, link] = (departure_time - t) * time.step_size
                                break
                            elif cvn_down[t2, link] == cvn:
                                found_cvn = True
                                travel_times[t, link] = (t2 - t) * time.step_size
                                break
                        if not found_cvn:
                            # vehicle unable to leave network during simulation
                            travel_times[t, link] = travel_times[t - 1, link]

                travel_times[t, link] = max(travel_times[t, link],
                                            np.float32((network.links.length[link] / network.links.v0[link])))
    else:
        raise NotImplementedError

    return travel_times


def _debug_plot(results, network: ILTMNetwork, time, title='None', toy_network=True):
    from dtapy.__init__ import current_network
    flows = cvn_to_flows(results.cvn_down)
    cur_queues = np.sum(results.cvn_up, axis=2) - np.sum(results.cvn_down, axis=2)  # current queues
    show_assignment(current_network, time, toy_network=toy_network, title=title, link_kwargs=
    {'cvn_up': results.cvn_up, 'cvn_down': results.cvn_down, 'vind': network.links.vf_index,
     'wind': network.links.vw_index, 'flows': flows, 'current_queues': cur_queues})
