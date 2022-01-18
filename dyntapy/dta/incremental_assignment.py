import numpy as np
from numba import njit

from dyntapy.demand import InternalDynamicDemand
from dyntapy.dta.aon import get_aon_route_choice, link_to_turn_costs_deterministic
from dyntapy.dta.deterministic import update_arrival_maps, get_turning_fractions
from dyntapy.dta.i_ltm import i_ltm
from dyntapy.dta.i_ltm_cls import ILTMState
from dyntapy.dta.i_ltm_setup import i_ltm_aon_setup
from dyntapy.dta.time import SimulationTime
from dyntapy.dta.travel_times import cvn_to_travel_times
from dyntapy.results import cvn_to_flows
from dyntapy.supply import Network
from dyntapy.utilities import _log


@njit(cache=True)
def incremental(
    network: Network, dynamic_demand: InternalDynamicDemand, time: SimulationTime
):
    iltm_state, network = i_ltm_aon_setup(network, time, dynamic_demand)
    incremental_loading(network, time, dynamic_demand, 20, iltm_state)
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
def incremental_loading(
    network: Network,
    time: SimulationTime,
    dynamic_demand: InternalDynamicDemand,
    K,
    iltm_state: ILTMState,
):
    """
    Parameters
    ----------
    network :
    time :
    dynamic_demand :
    k : number of increments

    Returns
    -------
    object of type RouteChoiceState
    with turning fractions as acquired through the incremental loading.
    """
    _log("setting up incremental loading", to_console=True)
    aon_state = get_aon_route_choice(network, time, dynamic_demand)
    for k in range(1, K + 1):
        _log("incremental loading k = " + str(k), to_console=True)
        # update demand such that the current slice of demand is added.
        if k == 1:
            demand_factor = np.float32(1 / K)
        else:
            demand_factor = np.float32(k / (k - 1))
        for demand in dynamic_demand.demands:
            demand.to_destinations.values = (
                demand.to_destinations.values * demand_factor
            )
            demand.to_origins.values = demand.to_origins.values * demand_factor
        # network loading and route choice are calculated
        i_ltm(network, dynamic_demand, iltm_state, time, aon_state.turning_fractions)
        link_costs = cvn_to_travel_times(
            cvn_up=np.sum(iltm_state.cvn_up, axis=2),
            cvn_down=np.sum(iltm_state.cvn_down, axis=2),
            time=time,
            network=network,
            con_down=iltm_state.con_down,
        )
        turn_costs = link_to_turn_costs_deterministic(
            link_costs,
            network.nodes.out_links,
            network.links.in_turns,
            network.tot_turns,
            time,
            network.links.link_type,
            aon_state.turning_fractions,
            network.links.length / network.links.free_speed,
            iltm_state.cvn_up,
            aon_state.turn_restrictions,
        )
        # from dyntapy.dta.core.debugging import plot_links_with_cost_changes
        # plot_links_with_cost_changes(aon_state.link_costs, link_costs, iltm_state)
        update_arrival_maps(
            network,
            time,
            dynamic_demand,
            aon_state.arrival_maps,
            aon_state.turn_costs,
            turn_costs,
            link_costs,
        )
        turning_fractions = get_turning_fractions(
            dynamic_demand, network, time, aon_state.arrival_maps, turn_costs
        )
        # smoothing turning fractions like you would in MSA
        aon_state.turning_fractions = np.add(
            (k) / (k + 1) * aon_state.turning_fractions, 1 / (k + 1) * turning_fractions
        )
    return aon_state
