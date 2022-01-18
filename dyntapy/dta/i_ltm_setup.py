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

from dyntapy.csr import F32CSRMatrix, UI8CSRMatrix, csr_prep
from dyntapy.demand import InternalDynamicDemand
from dyntapy.dta.i_ltm_cls import (
    ILTMLinks,
    ILTMNetwork,
    ILTMNodes,
    ILTMState,
)
from dyntapy.dta.time import SimulationTime
from dyntapy.supply import Network
from dyntapy.utilities import _log
from dyntapy.settings import parameters

v_wave_default = parameters.dynamic_assignment.network_loading.v_wave_default


@njit(cache=True)
def i_ltm_aon_setup(
    network: Network, time: SimulationTime, dynamic_demand: InternalDynamicDemand
):
    """
    Adding additional structures (arrays, sparse matrices) to the network for i_ltm
    and deterministic routing to work.
    """
    _log("setting up data structures for i_ltm", to_console=True)
    # link properties

    v_wave = np.full(network.tot_links, v_wave_default, dtype=np.float32)
    length = network.links.length
    v0 = network.links.free_speed
    capacity = network.links.capacity
    step_size = time.step_size
    vf_index = np.floor_divide((length / v0), step_size).astype(
        np.int32
    )  # 0 if a road can be traversed more than once during a time interval
    # int works as floor in matlab
    vf_ratio = (vf_index - (length / v0) / step_size + np.float32(1)).astype(
        np.float32
    )  # interpolation ratio
    vf_index = -vf_index - np.int32(
        1
    )  # -1 for roads that clear faster than the time step, -2 and upwards for slower
    # so if you want to know the sending flow for the current time step you have to
    # look vf_index time steps back
    vw_index = np.floor_divide((length / v_wave), step_size).astype(np.int32)
    vw_ratio = (vw_index - (length / v_wave) / step_size + 1).astype(np.float32)
    vw_index = -vw_index - np.int32(1)
    k_crit = (capacity / v0).astype(np.float32)
    k_jam = (capacity / v_wave + k_crit).astype(np.float32)
    for in_link, linktype in enumerate(network.links.link_type):
        if (
            linktype == 1 or linktype == -1
        ):  # we don't want queuing caused by access to connectors ..
            k_jam[in_link] = np.float32(1000000)
    k_jam[k_jam < 72] = np.float32(72)  # to be refined ..
    iltm_links = ILTMLinks(
        network.links, vf_index, vw_index, vf_ratio, vw_ratio, k_jam, k_crit, v_wave
    )

    # node properties
    # getting turn_based_in_links, turn_based_out_links for node model see jitclass
    # and dtapy/core/technical.md for details
    # assuming inLinks and OutLinks are topped at 256 for each Node

    tot_time_steps = time.tot_time_steps
    index_array_node_turns = np.empty((network.tot_turns, 2), dtype=np.uint32)
    val_in_links = np.empty(network.tot_turns, dtype=np.uint8)
    val_out_links = np.empty_like(val_in_links)
    turn_counter = 0
    val_in_link_cap = np.empty_like(capacity)
    val_out_link_cap = np.empty_like(capacity)
    _in_l_counter = 0

    _out_link_counter = 0
    for node in range(network.tot_nodes):
        # tot_in_links = network.network.nodes.tot_in_links[node]
        node_turns = np.where(network.turns.via_node == node)[0]
        node_in_links = network.nodes.in_links.get_nnz(node)
        node_out_links = network.nodes.out_links.get_nnz(node)
        for turn in node_turns:
            from_link = network.turns.from_link[turn]
            to_link = network.turns.to_link[turn]
            index_array_node_turns[turn_counter] = np.uint32(node), np.uint32(turn)
            val_in_links[turn_counter] = np.uint8(
                np.where(node_in_links == from_link)[0][0]
            )
            val_out_links[turn_counter] = np.uint8(
                np.where(node_out_links == to_link)[0][0]
            )
            turn_counter += 1
        for in_link in node_in_links:
            val_in_link_cap[_in_l_counter] = capacity[in_link]
            _in_l_counter += 1
        for out_link in node_out_links:
            val_out_link_cap[_out_link_counter] = capacity[out_link]
            _out_link_counter += 1
    turn_based_in_links = UI8CSRMatrix(
        *csr_prep(
            index_array_node_turns, val_in_links, (network.tot_nodes, network.tot_turns)
        )
    )
    turn_based_out_links = UI8CSRMatrix(
        *csr_prep(
            index_array_node_turns,
            val_out_links,
            (network.tot_nodes, network.tot_turns),
        )
    )
    in_link_cap = F32CSRMatrix(
        val_in_link_cap,
        network.nodes.in_links.col_index,
        network.nodes.in_links.row_index,
    )
    out_link_cap = F32CSRMatrix(
        val_out_link_cap,
        network.nodes.out_links.col_index,
        network.nodes.out_links.row_index,
    )
    iltm_nodes = ILTMNodes(
        network.nodes,
        turn_based_in_links,
        turn_based_out_links,
        in_link_cap,
        out_link_cap,
    )
    i_ltm_network = ILTMNetwork(network, iltm_links, iltm_nodes, network.turns)

    # attributes that share the same sparsity structure should have the same index
    # arrays and the underlying data should be stored with a shared matrix where each
    # row is an individual data array for a sparse matrix

    # setting up results object
    # cold start
    marg_comp = False
    tot_destinations = dynamic_demand.tot_active_destinations
    tot_links = i_ltm_network.tot_links
    tot_nodes = i_ltm_network.tot_nodes
    tot_turns = i_ltm_network.tot_turns
    costs = np.zeros(
        (tot_links, tot_time_steps), dtype=np.float32
    )  # order of arguments is changed here, for route choice
    # iterate over multiple time steps for a single link ..
    t0 = length / v0
    for t in range(tot_time_steps):
        costs[:, t] = t0
    cvn_up = np.zeros((tot_time_steps, tot_links, tot_destinations), dtype=np.float32)
    cvn_down = np.zeros_like(cvn_up)
    con_up = np.full((tot_time_steps, tot_links), False, dtype=np.bool_)
    con_down = np.full((tot_time_steps, tot_links), False, dtype=np.bool_)
    nodes_2_update = np.full((tot_time_steps, tot_nodes), False, dtype=np.bool_)
    for origin in dynamic_demand.all_active_origins:
        for t in range(tot_time_steps):
            nodes_2_update[t, origin] = True

    turning_fractions = np.zeros(
        (tot_time_steps, tot_turns, tot_destinations), dtype=np.float32
    )
    results = ILTMState(
        turning_fractions,
        cvn_up,
        cvn_down,
        con_up,
        con_down,
        marg_comp,
        nodes_2_update,
        costs,
    )
    return results, i_ltm_network
