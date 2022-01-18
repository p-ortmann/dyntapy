#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import njit, prange, uint32
from numba.typed import List

from dyntapy.csr import UI32CSRMatrix
from dyntapy.demand import InternalDynamicDemand
from dyntapy.dta.deterministic import RouteChoiceState, get_turning_fractions
from dyntapy.dta.dynamic_dijkstra import dijkstra
from dyntapy.dta.i_ltm import i_ltm
from dyntapy.dta.i_ltm_setup import i_ltm_aon_setup
from dyntapy.dta.time import SimulationTime
from dyntapy.dta.travel_times import cvn_to_travel_times
from dyntapy.results import cvn_to_flows
from dyntapy.settings import parameters
from dyntapy.supply import Network
from dyntapy.utilities import _log

restricted_turn_cost = parameters.dynamic_assignment.route_choice.restricted_turn_cost
route_choice_delta = parameters.dynamic_assignment.route_choice.delta_cost
route_choice_agg = parameters.dynamic_assignment.route_choice.aggregation
use_turn_delays = parameters.dynamic_assignment.network_loading.use_turn_delays


# TODO: test the @njit(parallel=True) option here


@njit(parallel=True, cache=True)
def link_to_turn_costs(
    link_costs: np.ndarray,
    out_links: UI32CSRMatrix,
    in_turns: UI32CSRMatrix,
    tot_turns,
    time: SimulationTime,
    turn_delays,
    use_turn_delays=use_turn_delays,
):
    # the turn costs are defined as the cost incurred on the from link + the turn
    # delay it does NOT include the travel time on the to_link of the turn the turn
    # delay itself is defined as the time it takes to make the turn on the node this
    # additional delay is not yet taken account in the propagation and by default 0
    # for all turns
    """
    calculates turn from link costs assuming no turn delays
    Parameters
    ----------
    out_links : csr, node x links
    link_costs : array, tot_time_steps x tot_links
    out_turns : csr, link x turns

    Returns
    -------
    turn_costs as an array, tot_time_steps x turns
    """
    tot_time_steps = link_costs.shape[0]
    turn_costs = np.zeros((tot_time_steps, tot_turns), dtype=np.float32)
    if use_turn_delays:
        for t in range(time.tot_time_steps):
            for node in prange(out_links.get_nnz_rows().size):
                # turn and link labelling follows the node labelling turns with the
                # same via node are labelled consecutively the same is usually true
                # for the outgoing links of a node (if it's not a connector)
                for to_link in out_links.get_nnz(node):
                    for turn, from_link in zip(
                        in_turns.get_nnz(to_link), in_turns.get_row(to_link)
                    ):
                        turn_costs[t, turn] += link_costs[t, from_link]
                        interpolation_fraction = (
                            link_costs[t, from_link] / time.step_size
                        )
                        if (
                            t + 2 + np.floor(interpolation_fraction)
                            > time.tot_time_steps
                        ):
                            turn_costs[t, turn] += turn_delays[-1, turn]
                        elif interpolation_fraction < 1:
                            turn_costs[t, turn] += np.interp(
                                interpolation_fraction,
                                [0, 1],
                                turn_delays[t : t + 2, turn],
                            )
                        else:
                            arrival_period = np.int32(np.floor(interpolation_fraction))
                            interpolation_fraction = (
                                interpolation_fraction - arrival_period
                            )
                            turn_costs[t, turn] += np.interp(
                                interpolation_fraction,
                                [0, 1],
                                turn_delays[arrival_period : arrival_period + 2, turn],
                            )
    else:
        # no interpolation needed
        for node in prange(out_links.get_nnz_rows().size):
            for to_link in out_links.get_nnz(node):
                for turn, from_link in zip(
                    in_turns.get_nnz(to_link), in_turns.get_row(to_link)
                ):
                    turn_costs[:, turn] = link_costs[:, from_link]

    return turn_costs


@njit(cache=True)
def link_to_turn_costs_deterministic(
    link_costs: np.ndarray,
    out_links: UI32CSRMatrix,
    in_turns: UI32CSRMatrix,
    tot_turns,
    time: SimulationTime,
    link_types,
    turning_fractions,
    ff_tt,
    cvn_up,
    turn_restrictions,
):
    tot_time_steps = link_costs.shape[0]
    turn_costs = np.zeros((tot_time_steps, tot_turns), dtype=np.float32)
    for t in range(time.tot_time_steps):
        for node in prange(out_links.get_nnz_rows().size):
            for to_link in out_links.get_nnz(node):
                for turn, from_link in zip(
                    in_turns.get_nnz(to_link), in_turns.get_row(to_link)
                ):
                    if turn_restrictions[turn]:
                        turn_costs[t, turn] = restricted_turn_cost
                    else:
                        turn_costs[t, turn] = link_costs[t, from_link]
    return turn_costs


@njit(cache=True)
def aon(network: Network, dynamic_demand: InternalDynamicDemand, time: SimulationTime):
    iltm_state, network = i_ltm_aon_setup(network, time, dynamic_demand)
    aon_state = get_aon_route_choice(network, time, dynamic_demand)
    i_ltm(network, dynamic_demand, iltm_state, time, aon_state.turning_fractions)
    link_costs = cvn_to_travel_times(
        cvn_up=np.sum(iltm_state.cvn_up, axis=2),
        cvn_down=np.sum(iltm_state.cvn_down, axis=2),
        time=time,
        network=network,
        con_down=iltm_state.con_down,
    )

    flows = cvn_to_flows(iltm_state.cvn_down)
    return flows, link_costs


@njit(cache=True)
def get_aon_route_choice(
    network: Network,
    time: SimulationTime,
    dynamic_demand: InternalDynamicDemand,
):
    _log("calculating aon route choice", to_console=True)
    free_flow_costs = network.links.length / network.links.free_speed
    costs = np.empty((time.tot_time_steps, network.tot_links), dtype=np.float32)
    for t in range(time.tot_time_steps):
        costs[t, :] = free_flow_costs
    turn_costs = link_to_turn_costs(
        costs,
        network.nodes.out_links,
        network.links.in_turns,
        network.tot_turns,
        time,
        np.empty((time.tot_time_steps, network.tot_turns)),
        use_turn_delays=False,
    )
    turn_restrictions = np.full(network.tot_turns, False, np.bool_)
    for turn, (from_node, to_node) in enumerate(
        zip(network.turns.from_node, network.turns.to_node)
    ):
        if from_node == to_node:
            turn_restrictions[turn] = True

    for turn in range(network.tot_turns):
        if turn_restrictions[turn]:
            turn_costs[:, turn] = restricted_turn_cost
    arrival_maps = init_arrival_maps(
        turn_costs,
        network.links.in_turns,
        dynamic_demand.all_active_destination_links,
        time.step_size,
        time.tot_time_steps,
        network.tot_links,
        List.empty_list(uint32),
        turn_restrictions,
    )  # since there are no u-turns centroid routing is
    # prevented by default.
    turning_fractions = get_turning_fractions(
        dynamic_demand, network, time, arrival_maps, turn_costs
    )
    return RouteChoiceState(
        costs, turn_costs, arrival_maps, turning_fractions, turn_restrictions
    )


@njit(cache=True)
def init_arrival_maps(
    costs,
    in_links,
    destinations,
    step_size,
    tot_time_steps,
    tot_nodes,
    centroids,
    turn_restrictions,
):
    _log("initializing arrival maps", to_console=True)
    # works for node link and link turn graph representation
    is_centroid = np.full(tot_nodes, False)
    for centroid in centroids:
        is_centroid[centroid] = True
    arrival_map = np.empty(
        (len(destinations), tot_time_steps, tot_nodes), dtype=np.float32
    )
    for _id, destination in enumerate(destinations):
        arrival_map[_id, 0, :] = dijkstra(
            costs[0, :], in_links, destination, tot_nodes, is_centroid
        )
        for t in range(1, tot_time_steps):
            arrival_map[_id, t, :] = (
                arrival_map[_id, 0, :] + t * step_size
            )  # init of all time steps with free flow vals
    return arrival_map
