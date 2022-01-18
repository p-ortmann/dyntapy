#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from collections import OrderedDict

import numpy as np
from numba.core.types import boolean, float32, int8, uint8, uint32
from numba.core.types.containers import ListType
from numba.experimental import jitclass

from dyntapy.csr import UI32CSRMatrix, f32csr_type, ui32csr_type

# We differentiate here between the generic Links Nodes and Turns object and more
# specialized objects that inherit from these classes Dynamic Traffic Assignment
# algorithms all use a base line of attributes that are kept in these baseline
# classes Algorithms like LTM need additional attributes - these are kept in objects
# that inherit from their respective class, so LTM has a special class for its links
# to store things like cvn .. the source code for the baseline classes is replicated
# both in jitclass decorated and undecorated (uncompiled) form because inheritance
# otherwise does not work at this point,
# see https://github.com/numba/numba/issues/1694 . adding a new assignment algorithm
# to this is meant to be made simpler this way, nothing changes in the assignment
# object one simply needs to define new extending network, turn, node and link
# objects as needed and write a setup file, as shown for i-ltm.

spec_link = [
    ("capacity", float32[:]),
    ("from_node", uint32[:]),
    ("to_node", uint32[:]),
    ("length", float32[:]),
    ("flow", float32[:, :]),
    ("out_turns", ui32csr_type),
    ("in_turns", ui32csr_type),
    ("v_wave", float32[:]),
    ("free_speed", float32[:]),
    ("type", int8[:]),
    ("lanes", uint8[:]),
    ("link_type", int8[:]),
]


# spec_link = OrderedDict(spec_link)
@jitclass(spec_link)
class Links(object):
    """
    A simple class that carries various arrays and CSR Matrices that have link ids as
    their index
    """

    def __init__(
        self,
        length,
        from_node,
        to_node,
        capacity,
        free_speed,
        out_turns,
        in_turns,
        lanes,
        link_type,
    ):
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.free_speed = free_speed
        self.out_turns = out_turns  # csr link x turns row is outgoing
        # links
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes
        self.link_type = link_type


class UncompiledLinks(object):
    """
    See Links class for docs
    """

    def __init__(
        self,
        length,
        from_node,
        to_node,
        capacity,
        free_speed,
        out_turns,
        in_turns,
        lanes,
        link_type,
    ):
        """
        See Links class for docs
        """
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.free_speed = free_speed
        self.out_turns = out_turns  # csr linkxlink row is outgoing turns
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes
        self.link_type = link_type


spec_node = [
    ("out_links", ui32csr_type),
    ("in_links", ui32csr_type),
    ("control_type", int8[:]),
    ("capacity", float32[:]),
    ("tot_out_links", uint32[:]),
    ("tot_in_links", uint32[:]),
    ("is_centroid", boolean[:]),
]


@jitclass(spec_node)
class Nodes(object):
    """
    A simple class that carries various arrays and CSR Matrices that have node ids as
    their index
    """

    def __init__(
        self,
        out_links: UI32CSRMatrix,
        in_links: UI32CSRMatrix,
        tot_out_links,
        tot_in_links,
        control_type,
        capacity,
        is_centroid,
    ):
        """
        out_links and in_links are sparse matrices in csr format that indicate
        connected links and their nodes
        both are nodes x links with f(i,link_id) = j and essentially carry the same
        information. There's duplication to
        avoid on-the-fly transformations.
        out_links is fromNode x Link and in_links toNode x Link in dim with toNode
        and fromNode as val, respectively.
        """
        self.out_links: UI32CSRMatrix = out_links
        self.in_links: UI32CSRMatrix = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        # self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity
        self.is_centroid = is_centroid


class UncompiledNodes(object):
    """
    See Nodes class for docs
    """

    def __init__(
        self,
        out_links: UI32CSRMatrix,
        in_links: UI32CSRMatrix,
        tot_out_links,
        tot_in_links,
        control_type,
        capacity,
        is_centroid,
    ):
        """
        See Nodes class for docs
        """
        self.out_links: UI32CSRMatrix = out_links
        self.in_links: UI32CSRMatrix = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        # self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity
        self.is_centroid = is_centroid


spec_turn = [
    ("db_restrictions", ui32csr_type),
    ("penalty", float32[:]),
    ("capacity", float32[:]),
    ("from_node", uint32[:]),
    ("to_node", uint32[:]),
    ("via_node", uint32[:]),
    ("from_link", uint32[:]),
    ("to_link", uint32[:]),
    ("turn_type", int8[:]),
]
spec_turn = OrderedDict(spec_turn)


@jitclass(spec_turn)
class Turns(object):
    """
    A simple class that carries various arrays and CSR Matrices that have turn ids as
    their index
    """

    # db_restrictions refer to destination based restrictions as used in recursive logit
    def __init__(
        self,
        penalty,
        capacity,
        from_node,
        via_node,
        to_node,
        from_link,
        to_link,
        turn_type,
    ):
        self.penalty = penalty
        self.capacity = capacity
        self.from_node = from_node
        self.to_node = to_node
        self.via_node = via_node
        self.from_link = from_link
        self.to_link = to_link
        self.turn_type = turn_type


try:
    spec_network = [
        ("links", Links.class_type.instance_type),
        ("nodes", Nodes.class_type.instance_type),
        ("turns", Turns.class_type.instance_type),
        ("tot_links", uint32),
        ("tot_nodes", uint32),
        ("tot_turns", uint32),
    ]
except Exception:
    # numba disabled
    spec_network = []


@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns):
        self.links: Links = links
        self.nodes: Nodes = nodes
        self.turns: Turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
        # TODO: add lookup tables for name to index


try:
    spec_uncompiled_network = [
        ("tot_links", uint32),
        ("tot_nodes", uint32),
        ("tot_turns", uint32),
    ]
except Exception:
    spec_uncompiled_network = []


class UncompiledNetwork(object):
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
        # TODO: add lookup tables for name to index
