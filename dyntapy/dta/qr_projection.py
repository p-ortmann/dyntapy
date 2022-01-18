#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import numpy as np
from numba import njit, prange

from dyntapy.demand import InternalDynamicDemand
from dyntapy.dta.i_ltm_cls import ILTMNetwork
from dyntapy.dta.time import SimulationTime
from dyntapy.settings import parameters

epsilon = parameters.dynamic_assignment.network_loading.epsilon
HISTORICAL_SHIFT_FACTOR = 0.1
TRANSLATION_FACTOR = 10
restricted_turn_cost = parameters.dynamic_assignment.route_choice.restricted_turn_cost


@njit(cache=True, parallel=True)
def qr_projection(
    cvn_down,
    arrival_map,
    turn_costs,
    network: ILTMNetwork,
    turning_fractions,
    dynamic_demand: InternalDynamicDemand,
    time: SimulationTime,
    k,
):
    cur_translation_factor = TRANSLATION_FACTOR / k
    max_local_out_turns = 0
    gec = np.full(
        time.tot_time_steps, np.finfo(np.float32).resolution, dtype=np.float32
    )
    links_to_update = np.full(
        (
            dynamic_demand.tot_active_destinations,
            network.tot_links,
            time.tot_time_steps,
        ),
        False,
    )
    links_to_check = np.full(network.tot_links, False)
    # gap value according to excess cost
    for link in range(network.tot_links):
        if network.links.out_turns.get_nnz(link).size > 1:
            links_to_check[link] = True
            max_local_out_turns = max(
                network.links.out_turns.get_nnz(link).size, max_local_out_turns
            )
            # only adding links with more than one outgoing turn
    links_to_check = np.nonzero(links_to_check)[0]
    # TODO: shuffle matrices ..
    for d in prange(dynamic_demand.tot_active_destinations):
        shift = np.zeros((time.tot_time_steps, network.tot_turns), dtype=np.float32)
        for t in range(time.tot_time_steps):
            for link in links_to_check:
                shortest_turns = np.full(max_local_out_turns, False)
                local_costs = np.full(max_local_out_turns, np.inf, dtype=np.float32)
                gec_local = 0
                min_cost = arrival_map[d, t, link]
                sum_shift = 0
                if min_cost >= restricted_turn_cost:
                    # links that cannot reach the destination don't need to be evaluated
                    continue
                for turn_id, (turn, out_link) in enumerate(
                    zip(
                        network.links.out_turns.get_nnz(link),
                        network.links.out_turns.get_row(link),
                    )
                ):
                    interpolation_fraction = turn_costs[t, turn] / time.step_size
                    if t + interpolation_fraction >= time.tot_time_steps - 1:
                        arrival = (
                            arrival_map[d, time.tot_time_steps - 1, out_link]
                            + turn_costs[t, turn]
                            - (time.tot_time_steps - 1 - t) * time.step_size
                        )
                    elif interpolation_fraction < 1:
                        arrival = (
                            arrival_map[d, t, out_link] * (1 - interpolation_fraction)
                            + interpolation_fraction * arrival_map[d, t + 1, out_link]
                        )
                    else:
                        t2 = np.int32(t + 1 + np.floor(interpolation_fraction))
                        interpolation_fraction = interpolation_fraction - np.floor(
                            interpolation_fraction
                        )
                        arrival = (
                            arrival_map[d, t2, out_link] * (interpolation_fraction)
                            + (1 - interpolation_fraction)
                            * arrival_map[d, t2 - 1, out_link]
                        )
                    local_costs[turn_id] = arrival
                    if arrival <= min_cost + 10 * np.finfo(np.float32).resolution:
                        # turn part of current shortest path tree
                        shortest_turns[turn_id] = True
                        if turning_fractions[d, t, turn] == 1:
                            break
                    else:
                        # only updating used turns
                        if turning_fractions[d, t, turn] > 0:
                            shift[t, turn] = max(
                                (min_cost - arrival) * cur_translation_factor, -0.1
                            )
                            # print(f'shifting away from {turn=} by {shift[t, turn]}')
                            # shift always < 0  because the turn is not on the epsilon-shortest-path-tree
                            # the dampening factor basically translates from the units of costs to a change in
                            # turning fractions, it's supposed to be updated as you progress through the simulation
                            # dynamic reduction of change based on convergence of earlier time steps
                            shift[t, turn] = max(
                                -turning_fractions[d, t, turn]
                                + np.finfo(np.float32).eps,
                                min(
                                    0,
                                    shift[t, turn]
                                    - HISTORICAL_SHIFT_FACTOR * np.sum(shift[:t, turn]),
                                ),
                            )
                            # the aptly named historical shift factor determines how much more we shift based on
                            # how much has been shifted in previous time slices
                            sum_shift = sum_shift + shift[t, turn]
                            if t > 0:
                                val = (
                                    (arrival - min_cost)
                                    * (cvn_down[t, link, d] - cvn_down[t - 1, link, d])
                                    * turning_fractions[d, t, turn]
                                )
                                # print(f'{link=} contributed {val} for {t=} for loading {turn=} ')
                                gec_local = (
                                    gec_local
                                    + (arrival - min_cost)
                                    * (cvn_down[t, link, d] - cvn_down[t - 1, link, d])
                                    * turning_fractions[d, t, turn]
                                )

                            else:
                                val = (arrival - min_cost) * (
                                    cvn_down[t, link, d] * turning_fractions[d, t, turn]
                                )
                                # print(f'{link=} contributed {val} for {t=} for loading {turn=} ')
                                gec_local = gec_local + (arrival - min_cost) * (
                                    cvn_down[t, link, d] * turning_fractions[d, t, turn]
                                )
                if np.abs(sum_shift) > 0:
                    # changes in route choice registered for current link
                    local_short_turns = np.nonzero(shortest_turns)[0]
                    if not local_short_turns.size > 0:
                        # if the min_cost stems from an arrival map
                        # that wasn't brought into consistency with the current cost,
                        # or there are very small differences.
                        # it shouldn't happen since we measure the difference to the previously computed arrival map
                        local_short_turns = np.nonzero(
                            local_costs == np.argmin(local_costs)
                        )[0]
                    ptr = 0
                    for local_turn_id, turn in enumerate(
                        network.links.out_turns.get_nnz(link)
                    ):
                        if ptr < local_short_turns.size:
                            if local_short_turns[ptr] == local_turn_id:
                                shift[t, turn] = shift[t, turn] + np.abs(
                                    sum_shift / local_short_turns.size
                                )
                                # the previously applied reductions on turns that are not on the shortest path tree
                                # are now evenly spread among the turns that are part of it, such that the sum of
                                # the turning fractions is still 1.
                                ptr += 1
                        turning_fractions[d, t, turn] = (
                            turning_fractions[d, t, turn] + shift[t, turn]
                        )
                gec[t] = gec[t] + gec_local
                if gec_local > np.finfo(np.float32).resolution:
                    links_to_update[d, link, t] = True
    return turning_fractions, gec, links_to_update
