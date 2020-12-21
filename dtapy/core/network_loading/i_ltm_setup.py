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
from dtapy.assignment import Assignment


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

    position_first_in = np.empty(assignment.number_of_nodes, dtype=np.uint32)
    position_first_out = np.empty(assignment.number_of_nodes, dtype=np.uint32)
    position_first_out[0] = 1
    position_first_in[0] = 1

    # turning fractions
    tot_time_steps=assignment.time.tot_time_steps
    incoming_links_tf_index =  np.empty(tot_time_steps, dtype=np.uint32)
    outgoing_links_tf_index = np.empty(tot_time_steps, dtype=np.uint32)
    nb_tf= np.empty(assignment.number_of_nodes, dtype=np.uint32)
    position_first_tf=np.empty(assignment.number_of_nodes, dtype=np.uint32)
    l_pos=np.uint32(1)
    for node in np.arange(assignment.number_of_nodes, dtype=np.uint32):
        node_turns=np.where(assignment.network.turns.via_node==node)[0]
        nb_tf[node]=np.uint32(len(node_turns))
        position_first_tf[node]=l_pos


    lpos = 1;
    for n=1:totNodes
    node_turns = np.where(assignment.)
    node_prop.nbTF(n) = length(node_turns);
    node_prop.positionFirstTF(n) = lpos;
    IncomingLinks = node_prop.IncomingLinksList(node_prop.positionFirstIn(n):node_prop.positionFirstIn(
        n) + node_prop.nbIncomingLinks(n) - 1);
    OutgoingLinks = node_prop.OutgoingLinksList(node_prop.positionFirstOut(n):node_prop.positionFirstOut(
        n) + node_prop.nbOutgoingLinks(n) - 1);
    for turn=node_turns'
    lin = find(IncomingLinks == Network.Turns.FromLink(turn));
    lout = find(OutgoingLinks == Network.Turns.ToLink(turn));
    node_prop.IncomingLinksTFindex(lpos) = lin;
    node_prop.OutgoingLinksTFindex(lpos) = lout;
    lpos = lpos + 1;


end
end

for node in np.arange(assignment.number_of_nodes)[1:]:
        position_first_in[node] = position_first_in[node - 1] + assignment.network.nodes.tot_in_links[node - 1]
        position_first_out[node] = position_first_out[node - 1] + assignment.network.nodes.tot_out_links[node - 1]
    iltm_nodes=ILTMNodes(assignment.network.nodes, position_first_in, position_first_out)
    assignment.network = ILTMNetwork(assignment.network, iltm_links, iltm_nodes,
                                     assignment.network.turns)

