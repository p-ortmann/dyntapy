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
from numba.typed import List

from dyntapy.csr import UI32CSRMatrix
from dyntapy.demand import InternalDynamicDemand, InternalStaticDemand
from dyntapy.dta.debugging import sum_of_turning_fractions, verify_assignment_state
from dyntapy.dta.i_ltm import i_ltm
from dyntapy.dta.i_ltm_setup import i_ltm_aon_setup
from dyntapy.dta.aon import link_to_turn_costs_deterministic
from dyntapy.dta.incremental_assignment import incremental_loading
from dyntapy.dta.deterministic import update_route_choice
from dyntapy.dta.time import SimulationTime
from dyntapy.dta.travel_times import cvn_to_travel_times
from dyntapy.results import cvn_to_flows
from dyntapy.settings import debugging, parameters
from dyntapy.supply import Network
from dyntapy.utilities import _log

max_iterations = parameters.dynamic_assignment.max_iterations
tot_itr_incremental = parameters.dynamic_assignment.network_loading.tot_itr_incremental
commodity_type = "destination"


def _i_ltm_aon(
    network: Network, dynamic_demand: InternalDynamicDemand, time: SimulationTime
):
    # network loading time and route choice time may differ in the future
    network_loading_time = time
    route_choice_time = time
    convergence = List()
    convergence.append(np.inf)
    iltm_state, network = i_ltm_aon_setup(network, network_loading_time, dynamic_demand)
    # incremental loading is used to trigger a reasonable initial state without
    # excessive spillback
    aon_state = incremental_loading(
        network, route_choice_time, dynamic_demand, tot_itr_incremental, iltm_state
    )
    link_costs = cvn_to_travel_times(
        cvn_up=np.sum(iltm_state.cvn_up, axis=2),
        cvn_down=np.sum(iltm_state.cvn_down, axis=2),
        time=network_loading_time,
        network=network,
        con_down=iltm_state.con_down,
    )
    k = 1
    converged = False
    if debugging:
        sum_of_turning_fractions(
            aon_state.turning_fractions,
            network.links.out_turns,
            network.links.link_type,
            network.turns.to_node,
            tot_centroids=dynamic_demand.tot_centroids,
        )
    while k < max_iterations and not converged:
        _log("calculating network state in iteration " + str(k), to_console=True)
        i_ltm(
            network,
            dynamic_demand,
            iltm_state,
            network_loading_time,
            aon_state.turning_fractions,
        )
        verify_assignment_state(
            network,
            aon_state.turning_fractions,
            iltm_state.cvn_up,
            iltm_state.cvn_down,
            dynamic_demand.tot_centroids,
        )
        link_costs = cvn_to_travel_times(
            cvn_up=np.sum(iltm_state.cvn_up, axis=2),
            cvn_down=np.sum(iltm_state.cvn_down, axis=2),
            time=network_loading_time,
            network=network,
            con_down=iltm_state.con_down,
        )
        turn_costs = link_to_turn_costs_deterministic(
            link_costs,
            network.nodes.out_links,
            network.links.in_turns,
            network.tot_turns,
            route_choice_time,
            network.links.link_type,
            aon_state.turning_fractions,
            network.links.length / network.links.free_speed,
            iltm_state.cvn_up,
            aon_state.turn_restrictions,
        )
        gec = update_route_choice(
            aon_state,
            link_costs,
            turn_costs,
            iltm_state.cvn_down,
            network,
            dynamic_demand,
            route_choice_time,
            k,
        )
        if k > 1:
            convergence.append(np.sum(gec))
            _log("new flows, gap is  : ")
            _log(gec)
            if np.all(gec < 0.001):
                converged = True
        k = k + 1
        sum_of_turning_fractions(
            aon_state.turning_fractions,
            network.links.out_turns,
            network.links.link_type,
            network.turns.to_node,
            tot_centroids=dynamic_demand.tot_centroids,
        )
    flows = cvn_to_flows(iltm_state.cvn_down)
    convergence_arr = np.empty(len(convergence))
    for _id, i in enumerate(convergence):
        convergence_arr[_id] = i
        return (
            link_costs,
            iltm_state.cvn_up,
            iltm_state.cvn_down,
            iltm_state.con_up,
            iltm_state.con_down,
            commodity_type,
            iltm_state.turning_fractions,
            aon_state.turn_costs,
            flows,
            dynamic_demand.all_active_origins,
            dynamic_demand.all_active_destinations,
        )


def i_ltm_aon(network, dynamic_demand, time):
    vals = _i_ltm_aon(network, dynamic_demand, time)
    attr = [
        "link_costs",
        "cvn_up",
        "cvn_down",
        "con_up",
        "con_down",
        "commodity_type",
        "turning_fractions",
        "turn_costs",
        "flows",
        "origins",
        "destinations",
    ]
    return {k: v for k, v in zip(attr, vals)}


@njit(cache=True)
def is_cost_converged(
    costs,
    flows,
    arrival_map,
    dynamic_demand: InternalDynamicDemand,
    step_size,
    out_links: UI32CSRMatrix,
    target_gap=parameters.dynamic_assignment.gap,
):
    """

    Parameters
    ----------
    out_links :
    step_size : np.float32, duration of a time step
    target_gap : np.float64, threshold for convergence
    costs : tot_time_steps x tot_links
    flows : tot_time_steps x tot_links
    arrival_map : tot_destinations x tot_time_steps x tot_nodes
    dynamic_demand : InternalDynamicDemand

    Returns
    -------
    Tuple(boolean, np.float64)

    """
    experienced_travel_times = np.sum(
        np.multiply(costs.astype(np.float64), flows.astype(np.float64))
    )
    shortest_path_travel_times = np.float64(0)
    for t in dynamic_demand.loading_time_steps:
        demand: InternalStaticDemand = dynamic_demand.get_demand(t)
        for origin in demand.to_destinations.get_nnz_rows():
            for flow, destination in zip(
                demand.to_destinations.get_row(origin),
                demand.to_destinations.get_nnz(origin),
            ):
                shortest_path_travel_times += flow * (
                    arrival_map[
                        np.flatnonzero(
                            dynamic_demand.all_active_destinations == destination
                        )[0],
                        t + 1,
                        out_links.get_nnz(origin)[0],
                    ]
                    - (t + 1) * step_size
                )
    gap_value = np.divide(experienced_travel_times, shortest_path_travel_times) - 1
    return gap_value < target_gap, gap_value
