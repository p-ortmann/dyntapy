#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import njit
from dtapy.core.network_loading.link_models.i_ltm_cls import ILTMNetwork
from dtapy.visualization import show_assignment
from dtapy.core.supply import Network
from dtapy.core.time import SimulationTime
from numba import njit, prange
from itertools import count


@njit(parallel=True)
def cvn_to_flows(cvn):
    """

    Parameters
    ----------
    cvn :

    Returns
    -------

    """
    tot_time_steps = cvn.shape[0]
    tot_links = cvn.shape[1]
    cvn = np.sum(cvn, axis=2)
    flows = np.zeros((tot_time_steps, tot_links), dtype=np.float32)
    flows[0, :] = cvn[0, :]
    for time in prange(1, tot_time_steps):
        flows[time, :] = np.abs(-cvn[time - 1, :] + cvn[time, :])
    return flows


def _debug_plot(results, network: ILTMNetwork, delta_change, time, title):
    from __init__ import current_network
    flows = cvn_to_flows(results.cvn_down)
    cur_queues = np.sum(results.cvn_up, axis=2) - np.sum(results.cvn_down, axis=2)  # current queues
    show_assignment(current_network, time, title=title, link_kwargs=
    {'cvn_up': results.cvn_up, 'cvn_down': results.cvn_down, 'vind': network.links.vf_index,
     'wind': network.links.vw_index, 'flows': flows, 'current_queues': cur_queues},
                    node_kwargs={'delta_change': delta_change},
                    highlight_nodes=[19, 90, 91, ])


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
    for t in prange(time.tot_time_steps):
        for link in prange(network.tot_links):
            non_zero_destinations = np.uint8(0)
            unloaded = np.any(cvn_down[t, link, :] > 0)
            for destination in range(tot_destinations):
                if cvn_down[t, link, destination] > 0:
                    unloaded = False
                    non_zero_destinations += np.uint8(1)
                    found_increment = False
                    time_counter = 0
                    while not found_increment:
                        if cvn_up[t, link, destination] < cvn_down[t, link, destination]:
                            found_increment = True
                            travel_times[t, link] += np.interp
            if unloaded:
                travel_times[t, link] = np.float32((network.links.length / network.links.v0) * 3600)

    # simTT = zeros(length(Links.Id), totT + 1);
    # % compute
    # the
    # simulated
    # travel
    # times
    # timeSteps = dt * (0:1:totT);
    # for l=1:length(Links.Length)
    # [down, iun] = unique(cvn_down(l,:), 'first');
    # if length(down) <= 1
    #     simTT(l,:)=Links.Length(l) / Links.V0Prt(l);
    # else
    #     simTT(l,:)=max(interp1(down, timeSteps(iun), cvn_up(l,:))-dt * (0:totT), Links.Length(l) / Links.V0Prt(l));
    #     simTT(l, cvn_up(l,:)-cvn_down(l,:) < 10 ^ -3)=Links.Length(l) / Links.V0Prt(l);
    #     for t=2:totT + 1
    #     simTT(l, t) = max(simTT(l, t), simTT(l, t - 1) - dt);
    # end
    # end
    #
    # % if length(down) <= 1
    #     % val = Links.Length(l) / Links.V0Prt(l) - dt * (0:totT);
    # % else
    # % val = interp1(down, timeSteps(iun), cvn_up(l,:))-dt * (0:totT);
    # % end
    # % nan_indx = isnan(val);
    # % simTT(l, not (nan_indx)) = max(val(not (nan_indx)), Links.Length(l) / Links.V0Prt(l));
    # % simTT(l, nan_indx) = Inf;
    # % for t=2:totT + 1
    # % simTT(l, t) = max(simTT(l, t), simTT(l, t - 1) - dt);
    # % end
    # end
    # end
