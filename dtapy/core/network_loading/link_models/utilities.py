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
def cvn_to_travel_times(cvn_up, cvn_down, time: SimulationTime, network: Network):
    """

    Parameters
    ----------
    cvn_up :
    cvn_down :
    time :
    links :

    Returns
    -------

    """
    tot_destinations = cvn_up.shape[2]
    travel_times = np.zeros((time.tot_time_steps, network.tot_links), dtype=np.float32)
    cvn_up_single_commodity = np.sum(cvn_up, axis=2)
    cvn_down_single_commodity = np.sum(cvn_down, axis=2)
    for t in prange(time.tot_time_steps):
        if t == 3:
            breakpoint()
        for link in prange(network.tot_links):
            loaded = cvn_up_single_commodity[t, link] > 0
            if network.links.link_type[link] == -1:
                loaded = False  # sink connectors always have free flow travel time
            else:
                if np.sum(cvn_up_single_commodity[t, link] - cvn_up_single_commodity[t - 1, link]) > 0 \
                        or (t == 0 and cvn_up_single_commodity[t, link] > 0):
                    loaded = True
                    found_cvn = False
                    for t2 in range(t, time.tot_time_steps, 1):
                        if cvn_down_single_commodity[t2, link] > cvn_up_single_commodity[t, link]:
                            found_cvn = True
                            if t2 == 0:
                                departure_time = \
                                    (cvn_down_single_commodity[t2, link] - cvn_up_single_commodity[
                                        t, link]) * 1 / (
                                        cvn_down_single_commodity[t2, link])
                            else:
                                departure_time = \
                                    (cvn_down_single_commodity[t2, link] - cvn_up_single_commodity[
                                        t, link]) * 1 / (
                                            cvn_down_single_commodity[t2, link] - cvn_down_single_commodity[
                                        t2 - 1, link])
                            departure_time = (departure_time + t2) * time.step_size * 3600
                            travel_times[t, link] = departure_time - t * 3600 * time.step_size
                        elif cvn_down_single_commodity[t2, link] == cvn_up_single_commodity[t, link]:
                            found_cvn = True
                            travel_times[t, link] = (t2 - t) * 3600 * time.step_size
                    if not found_cvn:
                        # vehicle unable to leave network during simulation
                        travel_times[t, link] = travel_times[t - 1, link]

            if not loaded:
                travel_times[t, link] = np.float32((network.links.length[link] / network.links.v0[link]) * 3600)
    return travel_times


def _debug_plot(results, network: ILTMNetwork, time, title='None', toy_network=True):
    from dtapy.__init__ import current_network
    flows = cvn_to_flows(results.cvn_down)
    cur_queues = np.sum(results.cvn_up, axis=2) - np.sum(results.cvn_down, axis=2)  # current queues
    show_assignment(current_network, time, toy_network=toy_network, title=title, link_kwargs=
    {'cvn_up': results.cvn_up, 'cvn_down': results.cvn_down, 'vind': network.links.vf_index,
     'wind': network.links.vw_index, 'flows': flows, 'current_queues': cur_queues})
