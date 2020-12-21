#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import Nodes, Links, Turns, ILTMLinks, ILTMNetwork, ILTMNodes
import numpy as np
from numba import int32, float32
from numba.typed import List
from dtapy.assignment import Assignment
from datastructures.csr import UI8CSRMatrix, csr_prep

def i_ltm_setup(assignment: Assignment):
    # link properties
    cvn_up = np.empty((assignment.number_of_time_steps, assignment.number_of_links), dtype=np.float32)
    cvn_down = np.empty((assignment.number_of_time_steps, assignment.number_of_links), dtype=np.float32)
    length = assignment.network.links.length
    v0 = assignment.network.links.v0
    step_size = assignment.time.step_size
    v_wave = assignment.network.links.v_wave
    vf_index = int32((length / v0) / step_size)  # uint works as floor in matlab
    vf_ratio = float32(vf_index - (length / v0) / step_size + 1)
    vf_index = -vf_index - 1
    vw_index = int32((length / v_wave) / step_size)
    vw_ratio = float32(vw_index - (length / v_wave) / step_size + 1)
    vw_index = -vw_index - 1
    iltm_links = ILTMLinks(assignment.network.links, cvn_up, cvn_down, vf_index, vw_index, vf_ratio, vw_ratio)

    # node properties
    # getting turn_based_in_links, turn_based_out_links for node model see jitclass and dtapy/core/technical.md for details
    # assuming inLinks and OutLinks are topped at 256 for each Node

    tot_time_steps = assignment.time.tot_time_steps
    index_array_in_links = np.empty((assignment.number_of_turns,2), dtype=np.uint32)
    index_array_out_links = np.empty_like(index_array_in_links)
    val_in_links = np.empty(assignment.number_of_turns, dtype=np.uint8)
    val_out_links = np.empty_like(val_in_links)
    turn_counter=0
    for node in np.arange(assignment.number_of_nodes, dtype=np.uint32):
        tot_in_links = assignment.network.nodes.tot_in_links[node]
        node_turns = np.where(assignment.network.turns.via_node == node)[0]
        node_in_links = assignment.network.nodes.in_links.get_nnz(node)
        node_out_links = assignment.network.nodes.out_links.get_nnz(node)
        for turn in node_turns:
            from_link = assignment.network.turns.from_link[turn]
            to_link = assignment.network.turns.to_link[turn]
            index_array_in_links[turn_counter]=node, from_link
            index_array_out_links[turn_counter]=node, to_link
            val_in_links[turn_counter]=np.uint8(np.where(node_in_links == from_link)[0])
            val_out_links[turn_counter]=np.uint8(np.where(node_out_links == to_link)[0])
            turn_counter+=1
    turn_based_in_links=UI8CSRMatrix(*csr_prep(index_array_in_links, val_in_links, (assignment.number_of_nodes, assignment.number_of_turns)))
    turn_based_out_links=UI8CSRMatrix(*csr_prep(index_array_out_links, val_out_links, (assignment.number_of_nodes, assignment.number_of_turns)))
    iltm_nodes=ILTMNodes(assignment.network.nodes, turn_based_in_links, turn_based_out_links)
    assignment.network = ILTMNetwork(assignment.network, iltm_links, iltm_nodes,
          assignment.network.turns)

