#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
"""

after initializing either a `dyntapy.assignments.StaticAssignment` or
`dyntapy.assignments.DynamicAssignment` we have access to a compiled
`dyntapy.supply.Network` object.
Alternatively, this can be build using `dyntapy.supply_data.build_network`

>>> network = dyntapy.supply_data.build_network(g)

The structure of this network object is described below.
It gives access to a Links, Nodes and Turns object such that one can easily retrieve
any network information rather intuitively.

For example, if one wanted to get the free flow travel times for all links,
simply query the underlying links object.

>>> free_flow_costs = network.links.length/network.links.free_speed

"""
from collections import OrderedDict

from numba.core.types import boolean, float32, int8, uint8, uint32
from numba.experimental import jitclass

from dyntapy.csr import UI32CSRMatrix, f32csr_type, ui32csr_type

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
    specifies internal Links object

    Parameters
    ----------
    length: numpy.ndarray
        float, 1D
    from_node: numpy.ndarray
        int, 1D
    to_node: numpy.ndarray
        int, 1D
    capacity: numpy.ndarray
        float, 1D
    free_speed: numpy.ndarray
        float, 1D
    out_turns: dyntapy.csr.UI32CSRMatrix
    in_turns: dyntapy.csr.UI32CSRMatrix
    lanes: numpy.ndarray
        int, 1D
    link_type: numpy.ndarray
        int, 1D

    Notes
    -----

    should not be initialized by the user, use dyntapy.supply_data.build_network

    `out_turns` and `in_turns` are sparse matrices in CSR format that indicate
    connected turns and their links.
    Both have the same shape (`network.tot_turns`, `network.tot_links`) with the
    indexes indicating the link_id and the values the to- and from_link, respectively.
    There's duplication to avoid on-the-fly transformations.
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
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes
        self.link_type = link_type


class _UncompiledLinks(object):
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
    ("x_coord", float32[:]),
    ("y_coord", float32[:]),
]


@jitclass(spec_node)
class Nodes(object):
    """
    specifies internal Nodes object

    Parameters
    ----------
    out_links: dyntapy.csr.UI32CSRMatrix
    in_links: dyntapy.csr.UI32CSRMatrix
    tot_out_links: numpy.ndarray
        int, 1D - number of outgoing links
    tot_in_links: numpy.ndarray
        int, 1D - number of outgoing links
    control_type: numpy.ndarray
        int, 1D
    capacity: numpy.ndarray
        float, 1D
    is_centroid: numpy.ndarray
        bool, 1D
    x_coord: numpy.ndarray
        float, 1D
    y_coord: numpy.ndarray
        float, 1D
    Notes
    -----

    should not be initialized by the user, use dyntapy.supply_data.build_network

    `out_links` and `in_links` are sparse matrices in CSR format that indicate
    connected links and their nodes.
    Both have the same shape (`network.tot_nodes`, `network.tot_links`) with the
    indexes indicating the link_id and the values the to- and from_node, respectively.
    There's duplication to avoid on-the-fly transformations.

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
        x_coord,
        y_coord,
    ):
        self.out_links = out_links
        self.in_links: UI32CSRMatrix = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        self.control_type = control_type
        self.capacity = capacity
        self.is_centroid = is_centroid
        self.x_coord = x_coord
        self.y_coord = y_coord


class _UncompiledNodes(object):
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
        x_coord,
        y_coord,
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
        self.x_coord = x_coord
        self.y_coord = y_coord


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

    specifies internal Turns object
    should not be initialized by the user, use dyntapy.supply_data.build_network

    Parameters
    ----------
    penalty: numpy.ndarray
        float, 1D
    capacity: numpy.ndarray
        float, 1D
    from_node: numpy.ndarray
        int, 1D
    via_node: numpy.ndarray
        int, 1D
    to_node: numpy.ndarray
        int, 1D
    from_link: numpy.ndarray
        int, 1D
    to_link: numpy.ndarray
        int, 1D
    turn_type: numpy.ndarray
        int, 1D

    Notes
    -----

    should not be initialized by the user, use dyntapy.supply_data.build_network

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
    """
    specifies internal Network object

    Parameters
    ----------
    links: Links
    nodes: Nodes
    turns: Turns

    Notes
    -----

    should be initialized with dyntapy.supply_data.build_network

    See Also
    --------
    dyntapy.supply_data.build_network
    dyntapy.supply.Links
    dyntapy.supply.Nodes
    dyntapy.supply.Turns


    """

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


class _UncompiledNetwork(object):
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
