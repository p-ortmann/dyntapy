#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import Nodes, Links, Turns, ILTMLinks, ILTMNetwork, ILTMNodes, ILTMResults
import numpy as np
from numba import int32, float32
from numba.typed import List
from dtapy.assignment import Assignment
from datastructures.csr import UI8CSRMatrix, csr_prep, F32CSRMatrix


def i_ltm_setup(assignment: Assignment):
    # link properties
    length = assignment.network.links.length
    v0 = assignment.network.links.v0
    capacity = assignment.network.links.capacity
    step_size = assignment.time.step_size
    v_wave = assignment.network.links.v_wave
    vf_index = int32((length / v0) / step_size)  # uint works as floor in matlab
    vf_ratio = float32(vf_index - (length / v0) / step_size + 1)
    vf_index = -vf_index - 1
    vw_index = int32((length / v_wave) / step_size)
    vw_ratio = float32(vw_index - (length / v_wave) / step_size + 1)
    vw_index = -vw_index - 1
    k_crit = capacity / v0
    k_jam = capacity / v_wave + k_crit

    iltm_links = ILTMLinks(assignment.network.links, vf_index, vw_index, vf_ratio, vw_ratio, k_jam,
                           k_crit)

    # node properties
    # getting turn_based_in_links, turn_based_out_links for node model see jitclass and dtapy/core/technical.md for details
    # assuming inLinks and OutLinks are topped at 256 for each Node

    tot_time_steps = assignment.time.tot_time_steps
    index_array_in_links = np.empty((assignment.tot_turns, 2), dtype=np.uint32)
    index_array_out_links = np.empty_like(index_array_in_links)
    val_in_links = np.empty(assignment.tot_turns, dtype=np.uint8)
    val_out_links = np.empty_like(val_in_links)
    turn_counter = 0
    for node in range(assignment.tot_nodes):
        tot_in_links = assignment.network.nodes.tot_in_links[node]
        node_turns = np.where(assignment.network.turns.via_node == node)[0]
        node_in_links = assignment.network.nodes.in_links.get_nnz(node)
        node_out_links = assignment.network.nodes.out_links.get_nnz(node)
        for turn in node_turns:
            from_link = assignment.network.turns.from_link[turn]
            to_link = assignment.network.turns.to_link[turn]
            index_array_in_links[turn_counter] = node, from_link
            index_array_out_links[turn_counter] = node, to_link
            val_in_links[turn_counter] = np.uint8(np.where(node_in_links == from_link)[0])
            val_out_links[turn_counter] = np.uint8(np.where(node_out_links == to_link)[0])
            turn_counter += 1
    turn_based_in_links = UI8CSRMatrix(
        *csr_prep(index_array_in_links, val_in_links, (assignment.tot_nodes, assignment.tot_turns)))
    turn_based_out_links = UI8CSRMatrix(
        *csr_prep(index_array_out_links, val_out_links, (assignment.tot_nodes, assignment.tot_turns)))
    iltm_nodes = ILTMNodes(assignment.network.nodes, turn_based_in_links, turn_based_out_links)
    assignment.network = ILTMNetwork(assignment.network, iltm_links, iltm_nodes,
                                     assignment.network.turns)

    # setting up results object
    if not assignment.results:
        # cold start
        marg_comp=False
        tot_origins = assignment.demand_simulation.tot_origins
        all_origins = assignment.demand_simulation.next
        tot_destinations = assignment.demand_simulation.tot_destinations
        tot_links = assignment.tot_links
        tot_nodes = assignment.tot_nodes
        tot_turns = assignment.tot_turns

        cvn_up = np.zeros((tot_links, tot_destinations, tot_time_steps), dtype=np.float32)
        cvn_down = np.empty_like(cvn_up)
        con_up = np.full((tot_links, tot_time_steps), False, dtype=np.bool_)
        con_down = np.full((tot_links, tot_time_steps), False, dtype=np.bool_)
        nodes_2_update = np.full((tot_nodes, tot_time_steps), False, dtype=np.bool_)

        # in matlab all are active all the time .. however in our case this is not necessary, we'll see if it causes
        # issues down the line ..
        #
        # for t in np.arange(tot_time_steps):
        #     for origin in assignment.demand_simulation.demands[t].get_nnz_rows():
        #         nodes_2_update[origin][t]=True
        # we stick with the implementation where all are active and see if we can reduce later.
        for origin in range(tot_origins):
            nodes_2_update[origin] = True

        turning_fractions = List()
        for _ in range(tot_time_steps):
            index_array_tf = np.column_stack((assignment.network.turns.via_node, np.arange(tot_turns, dtype=np.uint32)))
            values = np.full(tot_turns, 0.0, dtype=np.float32)
            turning_fractions.append(F32CSRMatrix(*csr_prep(index_array_tf, values, (tot_nodes, tot_turns))))
        assignment.results = ILTMResults(turning_fractions, cvn_up, cvn_down, con_up, con_down, marg_comp, nodes_2_update)

