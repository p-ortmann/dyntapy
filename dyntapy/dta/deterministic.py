#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import bool_, float32, float64, njit, prange
from numba.experimental import jitclass

from dyntapy.demand import InternalDynamicDemand
from dyntapy.dta.qr_projection import qr_projection
from dyntapy.dta.time import SimulationTime
from dyntapy.supply import Network
from dyntapy.utilities import _log
from dyntapy.settings import parameters

spec_rc_state = [
    ("link_costs", float32[:, :]),
    ("turn_costs", float32[:, :]),
    ("arrival_maps", float32[:, :, :]),
    ("turning_fractions", float64[:, :, :]),
    ("turn_restrictions", bool_[:]),
]
restricted_turn_cost = parameters.dynamic_assignment.route_choice.restricted_turn_cost
route_choice_delta = parameters.dynamic_assignment.route_choice.delta_cost
route_choice_agg = parameters.dynamic_assignment.route_choice.aggregation


@jitclass(spec_rc_state)
class RouteChoiceState(object):
    def __init__(
        self, link_costs, turn_costs, arrival_maps, turning_fractions, turn_restrictions
    ):
        """
        Parameters
        ----------
        link_costs : float32 array, time_steps x links
        arrival_maps : float32 array, destinations x time_steps x nodes
        """
        self.link_costs = link_costs
        self.turn_costs = turn_costs
        self.arrival_maps = arrival_maps
        self.turning_fractions = turning_fractions
        self.turn_restrictions = turn_restrictions


@njit(cache=True)
def update_route_choice(
    state,
    link_costs: np.ndarray,
    turn_costs: np.ndarray,
    cvn_down,
    network: Network,
    dynamic_demand: InternalDynamicDemand,
    time: SimulationTime,
    k: int,
    method="quasi-reduced-projection",
):
    """

    Parameters
    ----------
    state : RouteChoiceState
    turn_costs : time_steps x links
    network : Network
    dynamic_demand : InternalDynamicDemand
    time : SimulationTime
    k : int, number of iteration
    method : str


    """
    update_arrival_maps(
        network,
        time,
        dynamic_demand,
        state.arrival_maps,
        state.turn_costs,
        turn_costs,
        link_costs,
    )
    if method == "msa":
        # deterministic case non convergent, saw tooth pattern settles in ..
        turning_fractions = get_turning_fractions(
            dynamic_demand, network, time, state.arrival_maps, turn_costs
        )
        state.turning_fractions = np.add(
            (k - 1) / k * state.turning_fractions, 1 / k * turning_fractions
        )
        state.turn_costs = turn_costs
        gec = np.full(
            time.tot_time_steps, np.finfo(np.float32).resolution, dtype=np.float32
        )
    elif method == "quasi-reduced-projection":
        # deterministic approach of updating the turning fractions, see willem's
        # thesis chapter 4 for background should lead to smooth convergence
        _, gec, _ = qr_projection(
            cvn_down,
            state.arrival_maps,
            turn_costs,
            network,
            state.turning_fractions,
            dynamic_demand,
            time,
            k,
        )
        state.turn_costs = turn_costs
    else:
        raise NotImplementedError
    return gec


@njit(cache=True, parallel=True)
def update_arrival_maps(
    network: Network,
    time: SimulationTime,
    dynamic_demand: InternalDynamicDemand,
    arrival_maps,
    old_costs,
    new_costs,
    link_costs,
):
    _log("updating arrival maps", to_console=True)
    tot_time_steps = time.tot_time_steps
    from_link = network.turns.from_link
    out_turns = network.links.out_turns
    in_turns = network.links.in_turns
    all_destinations = dynamic_demand.all_active_destinations
    delta_costs = np.abs(new_costs - old_costs)
    links_2_update = np.full(
        (tot_time_steps, network.tot_links), False, dtype=np.bool_
    )  # nodes to be
    # activated for earlier time steps
    step_size = time.step_size
    turn_time = np.floor_divide(new_costs, step_size)
    interpolation_frac = np.divide(new_costs, step_size) - turn_time
    is_relevant_time_slice = np.full(tot_time_steps, False)
    # TODO: revisit structuring of the travel time arrays
    # could be worth while to copy and reverse the order depending on where you're at
    # in these loops ..
    # the following implementation closely follows the solution presented in
    # Himpe, Willem. "Integrated Algorithms for Repeated Dynamic Traffic Assignments
    # The Iterative Link Transmission Model with Equilibrium Assignment Procedure."(
    # 2016).refer to page 48, algorithm 6 for details.
    # only the time dependency grid has been added to reduce the
    # number of computational points being queried.
    for destination in prange(all_destinations.size):
        _log(" processing new destination")
        for t in range(tot_time_steps - 1, -1, -1):
            _log(
                "building map for destination "
                + str(destination)
                + " , now in time step "
                + str(t)
            )
            for turn, delta in np.ndenumerate(delta_costs[t, :]):
                # find all turns with changed travel times and add their from_links
                # to the list of links to be updated
                # u turn costs are set to infinity and do not change.
                if delta > route_choice_delta:
                    link = from_link[turn]
                    links_2_update[t, link] = True

            while np.any(links_2_update[t, :]):
                # _log('currently active nodes: '
                # + str(np.argwhere(nodes_2_update == True)))
                # going through all the nodes that need updating for the
                # current time step note that nodes_2_update changes dynamically as
                # we traverse the graph .. finding the node with the minimal arrival
                # time to the destination is meant to reduce the total nodes being
                # added to the nodes_2_update list

                min_dist = np.inf
                min_link = -1
                for link, active in enumerate(links_2_update[t, :]):
                    if active:
                        if arrival_maps[destination, t, link] <= min_dist:
                            min_link = link
                            min_dist = arrival_maps[destination, t, link]
                links_2_update[t, min_link] = False  # no longer considered
                # _log('deactivated node ' + str(min_node))
                new_dist = np.inf
                for out_link, turn in zip(
                    out_turns.get_row(min_link), out_turns.get_nnz(min_link)
                ):
                    if t + np.uint32(turn_time[t, turn]) >= tot_time_steps - 1:
                        dist = (
                            arrival_maps[destination, tot_time_steps - 1, out_link]
                            + new_costs[t, turn]
                            - (tot_time_steps - 1 - t) * step_size
                        )
                    else:
                        dist = (1 - interpolation_frac[t, turn]) * arrival_maps[
                            destination, t + np.uint32(turn_time[t, turn]), out_link
                        ] + interpolation_frac[t, turn] * arrival_maps[
                            destination, t + np.uint32(turn_time[t, turn]) + 1, out_link
                        ]
                    # _log(f'distance to {min_node} via
                    # out_link node {to_node[link]} is {dist} ')
                    if dist < new_dist:
                        new_dist = dist
                        assert new_dist > 0
                        assert new_dist < 1000

                # _log(f'result for node {min_node} written back? {np.abs(new_dist
                # - arrival_maps[destination, t, min_node]) > route_choice_delta}')
                if (
                    np.abs(new_dist - arrival_maps[destination, t, min_link])
                    > route_choice_delta
                ):
                    # new arrival time found
                    arrival_maps[destination, t, min_link] = new_dist
                    links_2_update[:t, min_link] = _find_relevant_time_slices(
                        t, link_costs[:, min_link], step_size, is_relevant_time_slice
                    )
                    for turn in in_turns.get_nnz(min_link):
                        links_2_update[t, from_link[turn]] = True
                        links_2_update[
                            :t, from_link[turn]
                        ] = _find_relevant_time_slices(
                            t,
                            link_costs[:, from_link[turn]],
                            step_size,
                            is_relevant_time_slice,
                        )


@njit(cache=True)
def get_turning_fractions(
    dynamic_demand: InternalDynamicDemand,
    network: Network,
    time: SimulationTime,
    arrival_maps,
    new_costs,
    departure_time_offset=route_choice_agg,
):
    """
    Calculates turning fractions taking into account closed turns. Deterministic
    procedure; a turn has a fraction of 1 if it is on the destination based shortest
    path tree, zero otherwise. Parameters ---------- network :
    numba.experimental.jitclass.boxing.Network dynamic_demand :
    numba.experimental.jitclass.boxing.InternalDynamicDemand state : AONState,
    see def departure_time_offset : float32 in [0,1] , indicates which departure time
    to consider in between two adjacent time intervals 0 indicates the very first
    vehicle is used to predict the choices of all in the interval, 0.5 the middle,
    and consequently 1 the last


    Returns
    -------

    """
    # calculation for the experienced travel times
    # _log('calculating arrival maps ')

    step_size = time.step_size
    turning_fractions = np.zeros(
        (
            dynamic_demand.tot_active_destinations,
            time.tot_time_steps,
            network.tot_turns,
        ),
        dtype=np.float64,
    )
    for dest_idx in range(dynamic_demand.all_active_destinations.size):
        # print(f'destination {dynamic_demand.all_active_destinations[dest_idx]}')
        for t in range(time.tot_time_steps):
            for link in range(network.tot_links):
                min_dist = np.inf
                min_turn = -1
                for out_turn, to_link in zip(
                    network.links.out_turns.get_nnz(link),
                    network.links.out_turns.get_row(link),
                ):
                    turn_time = np.floor(
                        departure_time_offset + new_costs[t, out_turn] / step_size
                    )
                    if t + np.uint32(turn_time) < time.tot_time_steps - 1:
                        interpolation_fraction = (
                            departure_time_offset
                            + new_costs[t, out_turn] / step_size
                            - turn_time
                        )
                        dist = (1 - interpolation_fraction) * arrival_maps[
                            dest_idx, t + np.uint32(turn_time), to_link
                        ] + interpolation_fraction * arrival_maps[
                            dest_idx, t + np.uint32(turn_time) + 1, to_link
                        ]
                    else:
                        dist = (
                            arrival_maps[dest_idx, time.tot_time_steps - 1, to_link]
                            + new_costs[t, out_turn]
                        )
                    if dist <= min_dist:
                        min_turn = out_turn
                        min_dist = dist
                if min_turn != -1:
                    turning_fractions[dest_idx, t, min_turn] = 1
    return turning_fractions


@njit()
def _find_relevant_time_slices(
    cur_interval, link_costs, step_size, is_relevant_time_slice
):
    """

    Parameters
    ----------
    cur_interval : int
    step_size : float
    is_relevant_time_slice : boolean array

    Returns
    -------
    boolean array
    """
    is_relevant_time_slice[:cur_interval] = False
    for t in range(cur_interval - 1, -1, -1):
        t1 = np.uint32(t + link_costs[t] / step_size)
        t2 = t1 + 1
        if cur_interval == t1 or cur_interval == t2:
            is_relevant_time_slice[t] = True
    return is_relevant_time_slice[:cur_interval]
