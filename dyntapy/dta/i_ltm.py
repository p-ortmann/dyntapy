#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import njit

from dyntapy.demand import InternalDynamicDemand, InternalStaticDemand
from dyntapy.dta.i_ltm_cls import ILTMNetwork, ILTMState
from dyntapy.dta.orca_nodel_model import orca_node_model as orca
from dyntapy.dta.time import SimulationTime
from dyntapy.settings import parameters
from dyntapy.utilities import _log

node_model_str = parameters.dynamic_assignment.network_loading.node_model
max_iterations = parameters.dynamic_assignment.network_loading.max_iterations
congestion_flow_delta_trigger = (
    parameters.dynamic_assignment.network_loading.cong_flow_delta_trigger
)
trigger_node_update_threshold = (
    parameters.dynamic_assignment.network_loading.trigger_node_update_threshold
)


@njit(cache=True)
def i_ltm(
    network: ILTMNetwork,
    dynamic_demand: InternalDynamicDemand,
    results: ILTMState,
    time: SimulationTime,
    turning_fractions,
):
    _log("starting i-ltm run", to_console=True)
    turning_fractions = turning_fractions.transpose(
        1, 2, 0
    ).copy()  # turning fractions in route choice typically get
    # (destinations, time, turn)
    tot_time_steps = time.tot_time_steps
    step_size = time.step_size

    tot_links = network.tot_links
    tot_nodes = network.tot_nodes
    tot_destinations = dynamic_demand.all_active_destinations.size
    max_out_links = np.max(network.nodes.tot_out_links)
    max_in_links = np.max(network.nodes.tot_in_links)
    in_links = network.nodes.in_links
    out_links = network.nodes.out_links

    to_node = network.links.to_node
    from_node = network.links.from_node
    cap = network.links.capacity
    kjm = network.links.k_jam
    length = network.links.length
    vind = network.links.vf_index
    vrt = network.links.vf_ratio
    wind = network.links.vw_index
    wrt = network.links.vw_ratio

    # local rename results properties
    # check for warm and cold starting is done in i_ltm_aon_setup.py

    nodes_2_update = results.nodes_2_update
    cvn_up, cvn_down = results.cvn_up, results.cvn_down
    con_up, con_down = results.con_up, results.con_down
    cvn_up[:, :, :] = 0
    cvn_down[:, :, :] = 0
    con_up[:, :] = False
    con_down[:, :] = False
    marg_comp = False

    # allocate memory to local variables some of these variables are filled
    # with different states, we first
    # calculate desired sending, receiving and turning flows and then consolidate
    # them in an update step after
    # acquiring the sending and receiving flow constraints from the node model.
    local_sending_flow = np.zeros(
        (max_in_links, tot_destinations), dtype=np.float32
    )  # what the in_links would like to send
    temp_local_sending_flow = np.zeros(
        (max_in_links, tot_destinations), dtype=np.float32
    )  # what they actually get to send
    tot_local_sending_flow = np.zeros(
        max_in_links, dtype=np.float32
    )  # taking into account capacity constraints
    # variables with local_x always concern structures for the node model
    # and it's surrounding links/turns
    local_receiving_flow = np.zeros((max_out_links, tot_destinations), dtype=np.float32)
    tot_local_receiving_flow = np.zeros(max_in_links, dtype=np.float32)

    local_turning_flows = np.zeros((max_in_links, max_out_links), dtype=np.float32)
    local_turning_fractions = np.zeros((max_in_links, max_out_links), dtype=np.float32)
    # Note: these two variables are link based and concern the entire graph
    tot_receiving_flow = np.zeros(tot_links, dtype=np.float32)
    sending_flow = np.zeros((tot_links, tot_destinations), dtype=np.float32)

    # forward implicit scheme
    # go sequentially over each time step
    tot_nodes_updates = np.uint32(0)  # tracking node updates over the entire DNL
    delta_change = np.zeros(tot_nodes, dtype=np.float32)

    for t in range(tot_time_steps):
        if not nodes_2_update[t, :].any():
            continue
        _log("time step " + str(t) + "in iltm")
        # sending and receiving flow to be initialized for all links
        receiving_flow_init = np.full(tot_links, True)
        sending_flow_init = np.full(tot_links, True)

        if dynamic_demand.is_loading(
            t
        ):  # check if any origins are sending flow into the network this time step
            _log("demand at time step  " + str(t) + " is loading ")
            current_demand: InternalStaticDemand = dynamic_demand.get_demand(t)
            t_id = np.argwhere(dynamic_demand.loading_time_steps == t)[0][0]
            __load_origin_flows(
                current_demand,
                nodes_2_update,
                t,
                t_id,
                cvn_up,
                cvn_down,
                temp_local_sending_flow,
                tot_nodes_updates,
                out_links,
                cap,
                step_size,
                con_up,
                vind,
                tot_time_steps,
                to_node,
                dynamic_demand.all_active_destinations,
            )
        for origin in dynamic_demand.all_active_origins:
            for connector in network.nodes.out_links.get_nnz(origin):
                cvn_up[t, connector, :] = np.maximum(
                    cvn_up[t - 1, connector, :], cvn_up[t, connector, :]
                )
                cvn_down[t, connector, :] = np.maximum(
                    cvn_down[t - 1, connector, :], cvn_down[t, connector, :]
                )

        first_intersection = (
            dynamic_demand.all_centroids.size
        )  # the first C nodes are centroids
        node_processing_order = np.arange(first_intersection, tot_nodes, dtype=np.int64)

        cur_nodes_2_update = len(
            node_processing_order
        )  # remaining nodes that need updating for the current iteration
        it = 0  # counter for current iteration
        while it < max_iterations and cur_nodes_2_update > 0:
            _log("new iterations in main loop, time step " + str(t))
            it = it + 1
            tot_nodes_updates = tot_nodes_updates + cur_nodes_2_update

            #  _______ main loops here, optimization crucial ______
            for node in node_processing_order[:cur_nodes_2_update]:
                local_in_links = in_links.get_nnz(node)
                local_out_links = out_links.get_nnz(node)
                tot_local_in_links = len(local_in_links)
                tot_local_out_links = len(local_out_links)
                _log(
                    "calculating node "
                    + str(node)
                    + " with "
                    + str(tot_local_out_links)
                    + " out_links and "
                    + str(tot_local_in_links)
                    + " in_links"
                )
                delta_change[node] = 0
                calc_sending_flows(
                    local_in_links,
                    cvn_up,
                    t,
                    cvn_down,
                    vind,
                    vrt,
                    cap,
                    sending_flow,
                    sending_flow_init,
                    local_sending_flow,
                    tot_local_sending_flow,
                    step_size,
                )
                calc_receiving_flows(
                    local_out_links,
                    wrt,
                    wind,
                    kjm,
                    length,
                    cap,
                    t,
                    tot_local_receiving_flow,
                    tot_receiving_flow,
                    cvn_down,
                    cvn_up,
                    receiving_flow_init,
                    step_size,
                )
                if len(local_out_links) == 1:
                    calc_turning_flows_merge(
                        local_in_links,
                        local_turning_flows,
                        local_sending_flow,
                        local_turning_fractions,
                    )
                else:
                    calc_turning_flows_general(
                        local_turning_fractions,
                        network.nodes.turn_based_in_links.get_row(node),
                        network.nodes.turn_based_out_links.get_row(node),
                        network.nodes.turn_based_out_links.get_nnz(node),
                        local_sending_flow,
                        local_turning_flows,
                        turning_fractions,
                        t,
                        len(local_in_links),
                        len(local_out_links),
                    )

                local_turning_capacity = np.full(
                    len(local_out_links) * len(local_in_links),
                    100000.0,
                    dtype=np.float32,
                )
                if node_model_str == "orca":
                    result_turning_flows = orca(
                        node,
                        tot_local_sending_flow[:tot_local_in_links],
                        local_turning_fractions[
                            :tot_local_in_links, :tot_local_out_links
                        ],
                        local_turning_flows[:tot_local_in_links, :tot_local_out_links],
                        tot_local_receiving_flow[:tot_local_out_links],
                        local_turning_capacity,
                        network.nodes.in_link_capacity.get_row(node),
                        len(local_in_links),
                        len(local_out_links),
                    )

                    _log("got past node model")
                else:
                    raise ValueError(
                        "node model " + str(node_model_str) + " not defined"
                    )
                update_cvns_and_delta_n(
                    result_turning_flows,
                    turning_fractions,
                    local_sending_flow,
                    temp_local_sending_flow,
                    local_receiving_flow,
                    len(local_out_links),
                    local_in_links,
                    local_out_links,
                    tot_local_sending_flow,
                    con_down,
                    network.nodes.in_link_capacity.get_row(node),
                    time.step_size,
                    t,
                    cvn_down,
                    wind,
                    wrt,
                    from_node,
                    nodes_2_update,
                    delta_change,
                    tot_time_steps,
                    network.links.out_turns,
                    cvn_up,
                    tot_receiving_flow,
                    con_up,
                    vind,
                    to_node,
                    vrt,
                    marg_comp,
                    node,
                    kjm,
                    length,
                    network.nodes.turn_based_out_links.get_row(node),
                    network.nodes.turn_based_out_links.get_nnz(node),
                )
                for centroid in dynamic_demand.all_centroids:
                    delta_change[centroid] = 0
            node_processing_order = np.ascontiguousarray(np.argsort(delta_change)[::-1])
            cur_nodes_2_update = np.uint32(
                np.sum(delta_change > trigger_node_update_threshold)
            )
            _log("remaining nodes 2 update in this t are:  " + str(cur_nodes_2_update))
        unload_destination_flows(
            nodes_2_update,
            dynamic_demand.all_active_destinations,
            network.nodes.in_links,
            tot_receiving_flow,
            t,
            temp_local_sending_flow,
            vrt,
            cvn_up,
            vind,
            cvn_down,
            tot_time_steps,
        )
    _log("iltm finished", to_console=True)
    results.marg_comp = True


@njit(cache=True)
def unload_destination_flows(
    nodes_2_update,
    destinations,
    in_links,
    tot_receiving_flow,
    t,
    tmp_sending_flow,
    vrt,
    cvn_up,
    vind,
    cvn_down,
    tot_time_steps,
):
    _log("unloading destination traffic")
    for destination in destinations:
        if nodes_2_update[t, destination]:
            for connector in in_links.get_nnz(destination):
                tot_receiving_flow[connector] = np.inf
                tmp_sending_flow[0, :] = (1 - vrt[connector]) * cvn_up[
                    max(0, t + vind[connector]), connector, :
                ] + vrt[connector] * cvn_up[
                    max(0, t + vind[connector] + 1), connector, :
                ]
                if (
                    np.sum(np.abs(tmp_sending_flow[0, :] - cvn_down[t, connector, :]))
                    > trigger_node_update_threshold
                ):
                    cvn_down[t, connector, :] = tmp_sending_flow[0, :]
                    nodes_2_update[
                        np.uint32(min(t + 1, tot_time_steps - 1)), destination
                    ] = True


@njit(cache=True)
def __load_origin_flows(
    current_demand,
    nodes_2_update,
    t,
    t_id,
    cvn_up,
    cvn_down,
    tmp_sending_flow,
    tot_nodes_updates,
    out_links,
    cap,
    step_size,
    con_up,
    vind,
    tot_time_steps,
    to_node,
    all_active_destinations,
):
    _log("loading origin flow")
    never_flow = True
    for origin in current_demand.origins:
        if nodes_2_update[t, origin]:
            tot_nodes_updates += 1
            for index, connector in np.ndenumerate(out_links.get_nnz(origin)):
                tmp_sending_flow[0, :] = 0
                for flow, destination in zip(
                    current_demand.to_destinations.get_row(origin),
                    current_demand.to_destinations.get_nnz(origin),
                ):
                    destination_id = np.argwhere(
                        all_active_destinations == destination
                    )[0, 0]
                    if flow > 0.001:
                        never_flow = False
                    tmp_sending_flow[0, destination_id] += flow

                if (
                    np.sum(
                        np.abs(
                            tmp_sending_flow[0, :]
                            - (cvn_up[t, connector, :])
                            - cvn_down[t, connector, :]
                        )
                    )
                    > trigger_node_update_threshold
                ):
                    nodes_2_update[
                        np.uint32(min(tot_time_steps - 1, t + 1)), origin
                    ] = True
                    cvn_up[t, connector, :] = (
                        tmp_sending_flow[0, :] + cvn_up[t - 1, connector, :]
                    )
                    if np.sum(tmp_sending_flow[0, :]) < cap[connector] * step_size:
                        con_up[t, connector] = False
                    else:
                        con_up[t, connector] = True
                        print("connector congestion on link")
                        print(str(connector))
                        print("origin:")
                        print(str(origin))
                        print("rounded sending flow")
                        print(str(int(np.sum(tmp_sending_flow[0, :]))))
                        raise Exception(
                            " connector congestion, overloaded local infrastructure"
                        )

                if vind[connector] == -1:
                    nodes_2_update[t, to_node[connector]] = True
                else:
                    try:
                        nodes_2_update[
                            t - vind[connector] - 1, to_node[connector]
                        ] = True
                        nodes_2_update[t - vind[connector], to_node[connector]] = True
                    except Exception:
                        assert t - vind[connector] > tot_time_steps - 1
                        if t - vind[connector] - 1 == tot_time_steps - 1:
                            nodes_2_update[
                                tot_time_steps - 1, to_node[connector]
                            ] = True
                        else:
                            _log(
                                "Simulation time period is too short for given demand."
                                " Not all vehicles are able to exit the network"
                            )
                            raise Exception(
                                "Simulation time period is too short for given demand."
                                " Not all vehicles are able to exit the network"
                            )
    if never_flow:
        raise Exception(
            "no flow was loaded for this timestep, StaticDemand object issues "
        )


@njit
def calc_sending_flows(
    local_in_links,
    cvn_up,
    t,
    cvn_down,
    vind,
    vrt,
    cap,
    sending_flow,
    sending_flow_init,
    local_sending_flow,
    tot_local_sending_flow,
    step_size,
):
    _log("calc sending flows")
    """

    Parameters
    ----------
    local_in_links :
    cvn_up :
    t :
    cvn_down :
    vind :
    vrt :
    cap :
    sending_flow :
    sending_flow_init :
    local_sending_flow :
    tot_local_sending_flow :
    step_size :

    Returns
    -------

    """
    for _id, link in enumerate(local_in_links):
        if sending_flow_init[link]:
            sending_flow_init[link] = False
            # cvn_up[t,link,:] =  np.maximum(cvn_up[t,link,:], cvn_up[t-1,link,:])
            if t + vind[link] < 0:
                sending_flow[link, :] = 0
            else:
                sending_flow[link, :] = (
                    cvn_up[t + vind[link], link, :] * (1 - vrt[link])
                    - cvn_down[t - 1, link, :]
                )
                if vind[link] < -1:
                    sending_flow[link, :] = (
                        sending_flow[link, :]
                        + vrt[link] * cvn_up[t + vind[link] + 1, link, :]
                    )

                # for all links with free flow travel times
                # larger than dt we interpolate
        local_sending_flow[_id, :] = sending_flow[link, :]
        if vind[link] == -1:
            local_sending_flow[_id, :] = (
                local_sending_flow[_id, :] + vrt[link] * cvn_up[t, link, :]
            )
            # if link can be traversed during a single dt
            # we consider part of the current inflow
        local_sending_flow[_id, :][
            local_sending_flow[_id, :] < 0
        ] = 0  # setting negative sending flows to 0
        over_capacity_ratio = np.sum(local_sending_flow[_id, :]) / (
            cap[link] * step_size
        )
        if over_capacity_ratio > 1:
            tot_local_sending_flow[_id] = cap[link] * step_size
            local_sending_flow[_id, :] = (
                local_sending_flow[_id, :] / over_capacity_ratio
            )
        else:
            tot_local_sending_flow[_id] = np.sum(local_sending_flow[_id, :])


@njit
def calc_receiving_flows(
    local_out_links,
    wrt,
    wind,
    kjm,
    length,
    cap,
    t,
    tot_local_receiving_flow,
    tot_receiving_flow,
    cvn_down,
    cvn_up,
    receiving_flow_init,
    step_size,
):
    _log("calc receiving flows")
    for out_id, link in enumerate(local_out_links):
        tot_local_receiving_flow[out_id] = 0
        if receiving_flow_init[link]:
            receiving_flow_init[link] = False
            tot_receiving_flow[link] = (
                np.sum(cvn_down[max(0, t + wind[link]), link, :]) * (1 - wrt[link])
                - np.sum(cvn_up[max(t - 1, 0), link, :])
                + kjm[link] * length[link]
            )
            if wind[link] < -1:
                tot_receiving_flow[link] = tot_receiving_flow[link] + wrt[
                    link
                ] * np.sum(cvn_down[max(0, t + wind[link] + 1), link, :])
                # TODO: check if this offset is correct
        # if rf_down_cvn_db[link]<np.float32(pow(10,-10)):
        #   raise ValueError('negative RF')
        tot_local_receiving_flow[out_id] = tot_receiving_flow[link]
        if wind[link] == -1:
            tot_local_receiving_flow[out_id] = tot_local_receiving_flow[out_id] + wrt[
                link
            ] * np.sum(cvn_down[t, link, :])
        if tot_local_receiving_flow[out_id] < 0:
            # print(f'negative rf ' + str(tot_local_receiving_flow[out_id]))
            tot_local_receiving_flow[out_id] = 0
        tot_local_receiving_flow[out_id] = min(
            cap[link] * step_size, tot_local_receiving_flow[out_id]
        )


@njit(cache=True)
def calc_turning_flows_general(
    local_turning_fractions,
    turn_in_links,
    turn_out_links,
    turns,
    local_sending_flow,
    local_turning_flows,
    turning_fractions,
    t,
    tot_in_links,
    tot_out_links,
):
    """
    calculate the local turning flows for a general intersection

    Parameters
    ----------
    tot_out_links : int, number of out links of local node
    local_turning_fractions : array, dim>= local_in_links x local_out_links, row sum
    is 1 for row 0,..,k with k
    the number of in_links for the node
    turn_in_links : array, local in_link labels for all turns crossing the node
    turn_out_links : array, local out_link labels for all turns crossing the node
    turns : array, global turn ids of crossing turns
    local_sending_flow : array, dim>= local_in_links x tot_destinations
    local_turning_flows : array, dim>= local_in_links x local_out_links
    turning_fractions : array, dim tot_time_steps x tot_turns x destinations
    t : scalar, current time_step
    """
    _log("calc turning flows")
    local_turning_flows[:tot_in_links, :tot_out_links] = np.float32(0)
    # multiple outgoing links
    for in_id, out_id, turn in zip(turn_in_links, turn_out_links, turns):
        # in_link and out_link values here start at zero as they reference
        # the internal labelling for the node model
        local_turning_flows[in_id, out_id] = local_turning_flows[
            in_id, out_id
        ] + np.sum(local_sending_flow[in_id, :] * turning_fractions[t, turn, :])
    for in_id in range(tot_in_links):
        max_desired_out_flow = np.sum(
            local_sending_flow[in_id, :]
        )  # no capacity constraints considered yet
        if max_desired_out_flow == np.float32(0):
            for out_id in range(tot_out_links):
                local_turning_fractions[in_id, out_id] = np.float32(0)
        else:
            for out_id in range(tot_out_links):
                local_turning_fractions[in_id, out_id] = (
                    local_turning_flows[in_id, out_id] / max_desired_out_flow
                )


@njit(cache=True)
def calc_turning_flows_merge(
    local_in_links, local_turning_flows, local_sending_flow, local_turning_fractions
):
    # simple merge
    local_turning_fractions[:, 0] = 1
    for in_id in range(len(local_in_links)):
        local_turning_flows[in_id, 0] = np.sum(local_sending_flow[in_id, :])


@njit(cache=True)
def update_cvns_and_delta_n(
    result_turning_flows,
    turning_fractions,
    sending_flow,
    temp_sending_flow,
    receiving_flow,
    tot_out_links,
    local_in_links,
    local_out_links,
    tot_local_sending_flow,
    con_down,
    in_link_capacity,
    time_step,
    t,
    cvn_down,
    wind,
    wrt,
    from_node,
    nodes_2_update,
    delta_change,
    tot_time_steps,
    out_turns,
    cvn_up,
    total_receiving_flow,
    con_up,
    vind,
    to_nodes,
    vrt,
    marg_comp,
    node,
    k_jam,
    length,
    turn_based_out_links,
    node_turns,
):
    _log("updating cvns")
    update_node = False
    result_tot_sending_flow = np.sum(result_turning_flows, axis=1)
    receiving_flow[:tot_out_links, :] = 0
    for in_id, in_link in enumerate(local_in_links):
        # calculate sending flows for all in_links with fixed reduction factors on
        # the sending
        # flows for all destinations
        # based on the constraints imposed by the node model
        # check for all of the in_links if any of their cvns change significantly
        # enough to trigger
        # recomputation of their tail nodes at backward time (spillback).
        if tot_local_sending_flow[in_id] > 0:
            if (
                result_tot_sending_flow[in_id]
                < tot_local_sending_flow[in_id] - congestion_flow_delta_trigger
            ):
                con_down[t, in_link] = True
                temp_sending_flow[in_id, :] = (
                    sending_flow[in_id, :]
                    / np.sum(sending_flow[in_id, :])
                    * result_tot_sending_flow[in_id]
                )
                # Note to self: this is where FIFO violations occur and where cars
                # are cut into pieces
            else:
                con_down[t, in_link] = False
                temp_sending_flow[in_id, :] = sending_flow[in_id, :]
            if t == 0:
                update_in_link = (
                    np.sum(
                        np.abs(cvn_down[t, in_link, :] - temp_sending_flow[in_id, :])
                    )
                    > trigger_node_update_threshold
                )
            else:
                update_in_link = (
                    np.sum(
                        np.abs(
                            cvn_down[t, in_link, :]
                            - (
                                cvn_down[t - 1, in_link, :]
                                + temp_sending_flow[in_id, :]
                            )
                        )
                    )
                    > trigger_node_update_threshold
                )

            update_node = update_in_link or update_node
            if update_in_link:
                if wind[in_link] == -1:
                    nodes_2_update[t, from_node[in_link]] = True
                    if t > 0:
                        delta_change[from_node[in_link]] = delta_change[
                            from_node[in_link]
                        ] + wrt[in_link] * np.sum(
                            np.abs(
                                cvn_down[t, in_link, :]
                                - (
                                    cvn_down[t - 1, in_link, :]
                                    + temp_sending_flow[in_id, :]
                                )
                            )
                        )
                    if t == 0:
                        delta_change[from_node[in_link]] = delta_change[
                            from_node[in_link]
                        ] = delta_change[from_node[in_link]] + wrt[in_link] * np.sum(
                            np.abs(
                                cvn_down[t, in_link, :] - temp_sending_flow[in_id, :]
                            )
                        )

                else:
                    nodes_2_update[
                        np.uint32(min(tot_time_steps - 1, t - wind[in_link])) - 1,
                        from_node[in_link],
                    ] = True
                    nodes_2_update[
                        np.uint32(min(tot_time_steps - 1, t - wind[in_link])),
                        from_node[in_link],
                    ] = True

            active_destinations = np.argwhere(sending_flow[in_id, :] > 0).flatten()

            if tot_out_links == 1:
                for destination in active_destinations:
                    receiving_flow[np.uint32(0), destination] = (
                        receiving_flow[np.uint32(0), destination]
                        + temp_sending_flow[in_id, destination]
                    )
            else:
                for destination in active_destinations:
                    for turn in out_turns.get_nnz(in_link):
                        local_turn_id = np.argwhere(node_turns == turn)[0][0]
                        out_id = turn_based_out_links[local_turn_id]
                        receiving_flow[out_id, destination] = (
                            receiving_flow[out_id, destination]
                            + temp_sending_flow[in_id, destination]
                            * turning_fractions[t, turn, destination]
                        )

        else:
            # no flow on this link
            temp_sending_flow[in_id, :] = 0
            con_down[t, in_link] = False

    for out_id, out_link in enumerate(local_out_links):
        if t == 0:
            update_out_link = (
                np.sum(np.abs(cvn_up[t, out_link, :] - receiving_flow[out_id, :]))
                > trigger_node_update_threshold
            )
        else:
            update_out_link = (
                np.sum(
                    np.abs(
                        cvn_up[t, out_link, :]
                        - (cvn_up[t - 1, out_link, :] + receiving_flow[out_id, :])
                    )
                )
                > trigger_node_update_threshold
            )
        update_node = update_out_link or update_node

        if np.sum(receiving_flow[out_id, :]) > total_receiving_flow[out_id]:
            con_up[t, out_link] = True
        else:
            con_up[t, out_link] = False
        if update_out_link:
            if vind[out_link] == -1:
                nodes_2_update[t, to_nodes[out_link]] = True
                if t == 0:
                    delta_change[to_nodes[out_link]] = delta_change[
                        to_nodes[out_link]
                    ] + vrt[out_link] * np.sum(
                        np.abs(cvn_up[t, out_link, :] - receiving_flow[out_id, :])
                    )
                if t > 0:
                    delta_change[to_nodes[out_link]] = delta_change[
                        to_nodes[out_link]
                    ] = delta_change[to_nodes[out_link]] + vrt[out_link] * np.sum(
                        np.abs(
                            cvn_up[t, out_link, :]
                            - (cvn_up[t - 1, out_link, :] + receiving_flow[out_id, :])
                        )
                    )
                # print(f'{node=} activated node {to_nodes[out_link]} for {t=} with {
                # delta_change[to_nodes[out_link]]=}')
            else:
                nodes_2_update[
                    np.uint32(min(t - vind[out_link] - 1, tot_time_steps - 1)),
                    to_nodes[out_link],
                ] = True
                nodes_2_update[
                    np.uint32(min(t - vind[out_link], tot_time_steps - 1)),
                    to_nodes[out_link],
                ] = True

        if marg_comp:
            potential_destinations = (
                receiving_flow[out_id] <= trigger_node_update_threshold
            )
            if np.any(potential_destinations):
                threshold = 0
                for destination in np.argwhere(potential_destinations)[0]:
                    threshold += np.abs(
                        cvn_up[t, out_link, destination]
                        - (
                            cvn_up[t - 1, out_link, destination]
                            + receiving_flow[out_id, destination]
                        )
                    )
                if threshold > trigger_node_update_threshold:
                    nodes_2_update[t, to_nodes[out_link]] = True
                    nodes_2_update[
                        np.uint32(min(tot_time_steps - 1, t)), to_nodes[out_link]
                    ] = True
                    delta_change[to_nodes[out_link]] = (
                        delta_change[to_nodes[out_link]]
                        - time_step * 1 / vind[out_link]
                    )
    if update_node:
        # print(f'node {node} change significant, updating cvns')
        nodes_2_update[np.uint32(min(tot_time_steps - 1, t + 1)), node] = True
    for in_id, in_link in enumerate(local_in_links):
        if t != 0:
            cvn_down[t, in_link, :] = (
                cvn_down[t - 1, in_link, :] + temp_sending_flow[in_id, :]
            )

        else:
            if np.sum(temp_sending_flow[in_id, :]) > 0:
                cvn_down[t, in_link, :] = temp_sending_flow[in_id, :]
    for out_id, out_link in enumerate(local_out_links):
        if t != 0:
            cvn_up[t, out_link, :] = (
                cvn_up[t - 1, out_link, :] + receiving_flow[out_id, :]
            )
        else:
            if np.sum(receiving_flow[out_id, :]) > 0:
                cvn_up[t, out_link, :] = receiving_flow[out_id, :]
