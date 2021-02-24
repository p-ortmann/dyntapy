#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from core import ILTMNetwork, ILTMNodes, ILTMState, ILTMLinks
import numpy as np
from numba import int32, float32
from assignment import Assignment
from datastructures.csr import UI8CSRMatrix, csr_prep, F32CSRMatrix






def i_ltm_setup(assignment: Assignment):
    # link properties
    length = assignment._network.links.length
    v0 = assignment._network.links.v0
    capacity = assignment._network.links.capacity
    step_size = assignment.time.step_size
    v_wave = assignment._network.links.v_wave
    vf_index = int32((length / v0) / step_size)  # uint works as floor in matlab
    vf_ratio = float32(vf_index - (length / v0) / step_size + 1)
    vf_index = -vf_index - 1
    vw_index = int32((length / v_wave) / step_size)
    vw_ratio = float32(vw_index - (length / v_wave) / step_size + 1)
    vw_index = -vw_index - 1
    k_crit = capacity / v0
    k_jam = capacity / v_wave + k_crit
    iltm_links = ILTMLinks(assignment._network.links, vf_index, vw_index, vf_ratio, vw_ratio, k_jam,
                           k_crit)

    # node properties
    # getting turn_based_in_links, turn_based_out_links for node model see jitclass and dtapy/core/technical.md for details
    # assuming inLinks and OutLinks are topped at 256 for each Node

    tot_time_steps = assignment.time.tot_time_steps
    index_array_node_turns = np.empty((assignment.tot_turns, 2), dtype=np.uint32)
    val_in_links = np.empty(assignment.tot_turns, dtype=np.uint8)
    val_out_links = np.empty_like(val_in_links)
    turn_counter = 0
    val_in_link_cap = np.empty_like(capacity)
    val_out_link_cap = np.empty_like(capacity)
    _in_l_counter = 0

    _out_link_counter = 0
    for node in range(assignment.tot_nodes):
        # tot_in_links = assignment.network.nodes.tot_in_links[node]
        node_turns = np.where(assignment._network.turns.via_node == node)[0]
        node_in_links = assignment._network.nodes.in_links.get_nnz(node)
        node_out_links = assignment._network.nodes.out_links.get_nnz(node)
        for turn in node_turns:
            from_link = assignment._network.turns.from_link[turn]
            to_link = assignment._network.turns.to_link[turn]
            index_array_node_turns[turn_counter] = node, turn_counter
            val_in_links[turn_counter] = np.uint8(np.where(node_in_links == from_link)[0])
            val_out_links[turn_counter] = np.uint8(np.where(node_out_links == to_link)[0])
            turn_counter += 1
    turn_based_in_links = UI8CSRMatrix(
        *csr_prep(index_array_node_turns, val_in_links, (assignment.tot_nodes, assignment.tot_turns)))
    turn_based_out_links = UI8CSRMatrix(
        *csr_prep(index_array_node_turns, val_out_links, (assignment.tot_nodes, assignment.tot_turns)))
    val_in_link_cap[_in_l_counter:_in_l_counter + len(node_in_links)] = capacity[node_in_links]
    val_out_link_cap[_out_link_counter:_out_link_counter + len(node_in_links)] = capacity[node_out_links]
    in_link_cap = F32CSRMatrix(val_in_link_cap, assignment._network.nodes.in_links._col_index,
                               assignment._network.nodes.in_links._row_index)
    out_link_cap = F32CSRMatrix(val_out_link_cap, assignment._network.nodes.out_links._col_index,
                                assignment._network.nodes.out_links._row_index)
    iltm_nodes = ILTMNodes(assignment._network.nodes, turn_based_in_links, turn_based_out_links, in_link_cap,
                           out_link_cap)
    assignment._network = ILTMNetwork(assignment._network, iltm_links, iltm_nodes,
                                      assignment._network.turns)

    # attributes that share the same sparsity structure should have the same index arrays and the underlying data
    # should be stored with a shared matrix where each row is an individual data array for a sparse matrix

    # setting up results object
    if not assignment.results:
        # cold start
        marg_comp = False
        tot_origins = assignment._dynamic_demand.tot_origins
        all_origins = assignment._dynamic_demand.next
        tot_destinations = assignment._dynamic_demand.tot_destinations
        tot_links = assignment.tot_links
        tot_nodes = assignment.tot_nodes
        tot_turns = assignment.tot_turns
        costs = np.empty(( tot_links, tot_time_steps), dtype=np.float32) #order of arguments is changed here, for route choice
        # iterate over multiple time steps for a single link ..
        t0= length/v0
        for t in range(tot_time_steps):
            costs[:, t] = t0
        cvn_up = np.zeros((tot_time_steps, tot_links, tot_destinations), dtype=np.float32)
        cvn_down = np.empty_like(cvn_up)
        con_up = np.full((tot_time_steps, tot_links), False, dtype=np.bool_)
        con_down = np.full((tot_time_steps, tot_links), False, dtype=np.bool_)
        nodes_2_update = np.full((tot_time_steps, tot_nodes), False, dtype=np.bool_)

        # in matlab all are active all the time .. however in our case this is not necessary, we'll see if it causes
        # issues down the line ..
        #
        # for t in np.arange(tot_time_steps):
        #     for origin in assignment.demand_simulation.demands[t].get_nnz_rows():
        #         nodes_2_update[origin][t]=True
        # we stick with the implementation where all are active and see if we can reduce later.
        for origin in assignment._dynamic_demand.all_origins:
            nodes_2_update[origin] = True

        turning_fractions = np.zeros((tot_time_steps, tot_turns, tot_destinations))
        # may investigate use of sparse structure for turning fractions
        assignment.results = ILTMState(turning_fractions, cvn_up, cvn_down, con_up, con_down, marg_comp,
                                       nodes_2_update, costs)
