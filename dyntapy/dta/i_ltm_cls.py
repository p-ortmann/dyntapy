#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from numba import boolean, float32, int32
from numba.experimental import jitclass

from dyntapy.csr import f32csr_type, ui8csr_type
from dyntapy.supply import (
    Links,
    Network,
    Nodes,
    Turns,
    UncompiledLinks,
    UncompiledNetwork,
    UncompiledNodes,
    spec_link,
    spec_node,
    spec_uncompiled_network,
)

try:
    spec_iltm_node = [
        ("nodes", Nodes.class_type.instance_type),
        ("turn_based_in_links", ui8csr_type),
        ("turn_based_out_links", ui8csr_type),
        ("in_link_capacity", f32csr_type),
        ("out_link_capacity", f32csr_type),
    ]
except Exception:
    spec_iltm_node = []


@jitclass(spec_node + spec_iltm_node)
class ILTMNodes(UncompiledNodes):
    __init__Nodes = UncompiledNodes.__init__

    def __init__(
        self,
        nodes,
        turn_based_in_links,
        turn_based_out_links,
        in_link_cap,
        out_link_cap,
    ):
        """

        Parameters
        ----------
        nodes : Nodes.class_type.instance_type, baseline node object
        turn_based_in_links : csr matrix node x turns
        turn_based_out_links : csr matrix node x turns
        in_link_cap : csr matrix node x links
        out_link_cap: csr matrix node x links

        the values of the turn_based - in and out_link csr matrices are the index
        of the corresponding sending and receiving flow vector that the node model gets,
         capacities are also given
        ordered by in- and out links, see technical.md.
        """
        self.__init__Nodes(
            nodes.out_links,
            nodes.in_links,
            nodes.tot_out_links,
            nodes.tot_in_links,
            nodes.control_type,
            nodes.capacity,
            nodes.is_centroid,
        )

        self.turn_based_in_links = turn_based_in_links
        self.turn_based_out_links = turn_based_out_links
        self.in_link_capacity = in_link_cap
        self.out_link_capacity = out_link_cap


try:
    spec_iltm_link = [
        ("links", Links.class_type.instance_type),
        ("k_jam", float32[:]),
        ("k_crit", float32[:]),
        ("vf_index", int32[:]),
        ("vf_ratio", float32[:]),
        ("vw_index", int32[:]),
        ("v_wave", float32[:]),
        ("vw_ratio", float32[:]),
    ]
except Exception:
    spec_iltm_link = []


@jitclass(spec_link + spec_iltm_link)
class ILTMLinks(UncompiledLinks):
    __init__Links = UncompiledLinks.__init__

    def __init__(
        self, links, vf_index, vw_index, vf_ratio, vw_ratio, k_jam, k_crit, v_wave
    ):
        self.__init__Links(
            links.length,
            links.from_node,
            links.to_node,
            links.capacity,
            links.free_speed,
            links.out_turns,
            links.in_turns,
            links.lanes,
            links.link_type,
        )
        self.v_wave = v_wave
        self.vf_index = vf_index
        self.vw_index = vw_index
        self.vf_ratio = vf_ratio
        self.vw_ratio = vw_ratio
        self.k_jam = k_jam
        self.k_crit = k_crit


spec_results = [
    ("turning_fractions", float32[:, :, :]),
    ("cvn_up", float32[:, :, :]),
    ("cvn_down", float32[:, :, :]),
    ("con_up", boolean[:, :]),
    ("con_down", boolean[:, :]),
    ("marg_comp", boolean),
    ("nodes_2_update", boolean[:, :]),
    ("costs", float32[:, :]),
]


@jitclass(spec_results)
class ILTMState(object):
    # in the future this may also be replaced and inherit from a parent Results class
    def __init__(
        self,
        turning_fractions,
        cvn_up,
        cvn_down,
        con_up,
        con_down,
        marg_comp,
        nodes_2_update,
        costs,
    ):
        self.turning_fractions = turning_fractions
        self.cvn_up = cvn_up
        self.cvn_down = cvn_down
        self.con_up = con_up
        self.con_down = con_down
        self.marg_comp = marg_comp
        self.nodes_2_update = nodes_2_update
        self.costs = costs


try:
    spec_iltm_network = [
        ("network", Network.class_type.instance_type),
        ("links", ILTMLinks.class_type.instance_type),
        ("nodes", ILTMNodes.class_type.instance_type),
        ("turns", Turns.class_type.instance_type),
    ]
except Exception:
    spec_iltm_network = []


@jitclass(spec_uncompiled_network + spec_iltm_network)
class ILTMNetwork(UncompiledNetwork):
    __init__Network = UncompiledNetwork.__init__

    def __init__(self, network, links, nodes, turns):
        self.__init__Network(
            links,
            nodes,
            turns,
            network.tot_links,
            network.tot_nodes,
            network.tot_turns,
        )
