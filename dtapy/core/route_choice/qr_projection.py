#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import numpy as np

from dtapy.core.network_loading.link_models.i_ltm_cls import ILTMNetwork
from dtapy.core.demand import InternalDynamicDemand
from dtapy.core.time import SimulationTime
from numba import prange
from numba.typed import List


def qr_projection(flows, arrival_map, turn_costs, network: ILTMNetwork, turning_fractions,
                  dynamic_demand: InternalDynamicDemand, time: SimulationTime, dampening_factor, dr, tot_sum):
    """

    Parameters
    ----------
    dampening_factor :
    time :
    cvn_up :
    cvn_down :
    arrival_map :
    turn_costs :
    nodes :
    links :
    dynamic_demand :
    dr :
    tot_sum :

    Returns
    -------

    """
    links_to_check = List()
    max_local_out_turns = 0

    for link in network.tot_links:
        if network.links.out_turns.get_nnz(link).size > 1:
            links_to_check.append(link)
            max_local_out_turns = max(network.links.out_turns.get_nnz(link).size, max_local_out_turns)
            # only adding links with more than one outgoing turn
    shortest_turns = np.full(max_local_out_turns,False)
    for d in prange(dynamic_demand.tot_active_destinations):
        shift = np.zeros((network.tot_turns, time.tot_time_steps), dtype=np.float32)
        local_costs = np.zeros(max_local_out_turns, dtype=np.float32)
        for t in range(time.tot_time_steps):
            for link in links_to_check:
                min_cost = arrival_map[d, t, link]
                for turn_id, (turn, out_link) in enumerate(zip(network.links.out_turns.get_nnz(link),
                                                               network.links.out_turns.get_row(link))):
                    interpolation_fraction = turn_costs[t, link] / time.step_size
                    if t + interpolation_fraction >= time.tot_time_steps:
                        arrival = arrival_map[d, time.tot_time_steps, out_link] + \
                                  interpolation_fraction * time.step_size
                    elif interpolation_fraction < 1:
                        arrival = arrival_map[d, t + 1, out_link] * (1 - interpolation_fraction) + \
                                  interpolation_fraction * arrival_map[d, t, out_link]
                    else:
                        t2 = t + 1 + np.floor(interpolation_fraction)
                        interpolation_fraction = interpolation_fraction - np.floor(interpolation_fraction)
                        arrival = arrival_map[d, t2, out_link] * (1 - interpolation_fraction) + \
                                  interpolation_fraction * arrival_map[d, t2 + 1, out_link]
                    local_costs[turn_id] = arrival
                    if arrival <= min_cost:
                        # turn part of current shortest path tree
                        gap = gap + min_cost *flows[t,link] * turning_fractions[t, turn,d]
                        if turning_fractions[t,turn, d]==1:
                            break
                        shortest_turns[turn_id]=True
                    else:
                        # only updating used turns
                        if turning_fractions[t,turn,d]>0:
                            shift[t,turn] = (min_cost-arrival) * dampening_factor
                            # dynamic reduction of change based on convergence of earlier time steps
                            shift[t,turn] = max(-turning_fractions[t,turn,d],
                                                min(0,shift[t,turn]-dr*np.sum(shift[:t, turn])))
                            sum_shift = sum_shift + shift[t,turn]




