#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from collections import OrderedDict
from dyntapy.datastructures.csr import UI32CSRMatrix, ui32csr_type, f32csr_type
from numba.core.types import float32, uint32, int8, uint8, boolean
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from numba import njit
import numpy as np

# We differentiate here between the generic Links Nodes and Turns object and more specialized objects that inherit
# from these classes Dynamic Traffic Assignment algorithms all use a base line of attributes that are kept in these
# baseline classes Algorithms like LTM need additional attributes - these are kept in objects that inherit from their
# respective class, so LTM has a special class for its links to store things like cvn .. the source code for the
# baseline classes is replicated both in jitclass decorated and undecorated (uncompiled) form because inheritance
# otherwise does not work at this point, see https://github.com/numba/numba/issues/1694 .
# adding a new assignment algorithm to this is meant to be made simpler this way,
# nothing changes in the assignment object
# one simply needs to define new extending network, turn, node and link objects as needed and write a setup file,
# as shown for i-ltm.

spec_link = [('capacity', float32[:]),
             ('from_node', uint32[:]),
             ('to_node', uint32[:]),
             ('length', float32[:]),
             ('flow', float32[:, :]),
             ('out_turns', ui32csr_type),
             ('in_turns', ui32csr_type),
             ('v_wave', float32[:]),
             ('v0', float32[:]),
             ('type', int8[:]),
             ('lanes', uint8[:]),
             ('link_type', int8[:])]


# spec_link = OrderedDict(spec_link)
@jitclass(spec_link)
class Links(object):
    """
    A simple class that carries various arrays and CSR Matrices that have link ids as their index
    """

    def __init__(self, length, from_node, to_node, capacity, v_wave, v0,
                 out_turns, in_turns, lanes, link_type):
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.v_wave = v_wave
        self.v0 = v0
        self.out_turns = out_turns  # csr link x turns row is outgoing links
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes
        self.link_type = link_type


class UncompiledLinks(object):
    """
   See Links class for docs
    """

    def __init__(self, length, from_node, to_node, capacity, v_wave, v0,
                 out_turns, in_turns, lanes, link_type):
        """
        See Links class for docs
        """
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.v_wave = v_wave
        self.v0 = v0
        self.out_turns = out_turns  # csr linkxlink row is outgoing turns
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes
        self.link_type = link_type


spec_node = [('out_links', ui32csr_type),
             ('in_links', ui32csr_type),
             ('control_type', int8[:]),
             ('capacity', float32[:]),
             ('tot_out_links', uint32[:]),
             ('tot_in_links', uint32[:])]

class Lanes(object):
    pass
    # placeholder to support raheleh's lane based node models
    # we will need to decompose each link into its associated lanes.
    # Links can be thought of as containers of lanes, in line with the GMNS spec.
    # for each lane we store a green time


@jitclass(spec_node)
class Nodes(object):
    """
    A simple class that carries various arrays and CSR Matrices that have node ids as their index
    """

    def __init__(self, out_links: UI32CSRMatrix, in_links: UI32CSRMatrix, tot_out_links, tot_in_links, control_type,
                 capacity):
        """
        out_links and in_links are sparse matrices in csr format that indicate connected links and their nodes
        both are nodes x links with f(i,link_id) = j and essentially carry the same information. There's duplication to
        avoid on-the-fly transformations.
        out_links is fromNode x Link and in_links toNode x Link in dim with toNode and fromNode as val, respectively.
        Parameters
        ----------
        out_links : I64CSRMatrix <uint32>
        in_links : I64CSRMatrix <uint32>
        """
        self.out_links: UI32CSRMatrix = out_links
        self.in_links: UI32CSRMatrix = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        # self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity


class UncompiledNodes(object):
    """
    See Nodes class for docs
    """

    def __init__(self, out_links: UI32CSRMatrix, in_links: UI32CSRMatrix, tot_out_links, tot_in_links, control_type,
                 capacity):
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


spec_turn = [('db_restrictions', ui32csr_type),
             ('t0', float32[:]),
             ('capacity', float32[:]),
             ('from_node', uint32[:]),
             ('to_node', uint32[:]),
             ('via_node', uint32[:]),
             ('from_link', uint32[:]),
             ('to_link', uint32[:]),
             ('type', int8[:])]
spec_turn = OrderedDict(spec_turn)


@jitclass(spec_turn)
class Turns(object):
    """
    A simple class that carries various arrays and CSR Matrices that have turn ids as their index
    """

    # db_restrictions refer to destination based restrictions as used in recursive logit
    def __init__(self, t0, capacity, from_node, via_node, to_node, from_link, to_link,
                 turn_type):
        self.t0 = t0
        self.capacity = capacity
        self.from_node = from_node
        self.to_node = to_node
        self.via_node = via_node
        self.from_link = from_link
        self.to_link = to_link
        self.type = turn_type

    def set_db_restrictions(self, db_restrictions):
        # happens in preprocessing after in initialization ..
        self.db_restrictions = db_restrictions


try:
    spec_static_event = [('events', f32csr_type),
                         ('attribute_id', uint32)]
    spec_static_event = OrderedDict(spec_static_event)
except Exception:
    spec_static_event =  []


@jitclass(spec_static_event)
class StaticEvent(object):
    # events is a CSR with timeslice x obj_ids = val
    def __init__(self, events, attribute_id):
        self.__events = events
        self.__attribute_id = attribute_id

    def get_ids(self, time_slice):
        return self.__events.get_nnz(time_slice)

    def get_values(self, time_slice):
        return self.__events.get_row(time_slice)

    def get_attr_id(self):
        return self.__attribute_id

try:
    spec_network = [('links', Links.class_type.instance_type),
                    ('nodes', Nodes.class_type.instance_type),
                    ('turns', Turns.class_type.instance_type),
                    ('static_events', ListType(StaticEvent.class_type.instance_type)),
                    ('tot_links', uint32),
                    ('tot_nodes', uint32),
                    ('tot_turns', uint32),
                    ('tot_connectors', uint32)]
except Exception:
    # numba disabled
    spec_network = []

@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns, tot_connectors):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
        self.tot_connectors = np.uint32(tot_connectors)
        # TODO: add lookup tables for name to index

    def set_static_events(self, list_static_events):
        self.static_events = list_static_events

    def update_event_changes(self, time_slice):
        for static_event in self.static_events:
            array = self.get_array(static_event.get_attr_id)
            for obj_id, val in zip(static_event.get_ids(time_slice), static_event.get_values(time_slice)):
                array[obj_id] = val

    def get_array(self, attr):
        if attr == 0:
            return self.links.capacity


# @jitclass(spec_network)
try:
    spec_uncompiled_network = [
        ('static_events', ListType(StaticEvent.class_type.instance_type)),
        ('tot_links', uint32),
        ('tot_nodes', uint32),
        ('tot_turns', uint32),
        ('tot_connectors', uint32)]
except Exception:
    spec_uncompiled_network=[]


class UncompiledNetwork(object):
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns, tot_connectors):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
        self.tot_connectors = tot_connectors
        # TODO: add lookup tables for name to index

    def set_static_events(self, list_static_events):
        self.static_events = list_static_events

    def update_event_changes(self, time_slice):
        for static_event in self.static_events:
            array = self.get_array(static_event.get_attr_id)
            for obj_id, val in zip(static_event.get_ids(time_slice), static_event.get_values(time_slice)):
                array[obj_id] = val

    def get_array(self, attr):
        if attr == 0:
            return self.links.capacity

# dt current time step
# T time horizon
# sets event changes in links turns nodes in accordance with what is registered in dynamic events and
# static events for that time period
# reduces overhead induced by these event mechanisms ..
# restricts controller to register changes in time horizon steps, is this

# def get_controller_response(self, dt, T, Results: Results):
# controllers are able to respond to results of the previous time step. LTM and the controller are assumed to be
# in sync in terms of their time steps.
# if smaller sampling of the controller is really crucial one can always opt for a smaller time step (at the expense
# of higher computational cost.
#    pass

# def create_dynamic_event_cls(type):
# if one is interested in creating multiple dynamic events, some of which that may be manipulating an integer array,
# we'd need a functions that handles the type definition of the specs and returns the appropriate class variant.
# rebuild the class below and return it ..

# return DynamicEvent

# how to handle dynamic changes, proposal: 3 column array with time stamp and value, temporary structutre for computations
# that reach into the past with +- k * dt(travel time on longest link), rewritten for each dt in the simulation
# alternative:
# register sparse matrix for each object, with obj_id x attribute_id=val as change in analysis TP, retrieval for specific time has
# overhead of checking the sparse structure ..
# if registered_events[link_capacity_id]:
#     def get_capacity(self, link_id, time=0):  # timing access and overhead through the sparse check,
# possibly different approach, depends on whether full array is needed for the 'time'
# or only a single entry
# alternative: add a check on whether event queue exists for this attribute (possibly hard, at compile time the object
# doesn't exist yet ...# - can functions be compiled differently based on this?
# TODO: test this design on compatibility with numba
# if time == 0:
#     return self.capacity[link_id]
# else:
#     if self.event_changes.get_nnz(link_id) != link_capacity_id:
#         return self.capacity[link_id]
#     else:
# interpolation ..
# return 0
# else:
#     def get_capacity(self, link_id, time=0): # this will get inlined by the jit compiler - no overhead .
#         return self.capacity[link_id]
# def get_var(self, link_id, attribute, time=0):
#     pass
# tup_type = UniTuple(float32, 3)
# spec_dynamic_events = [('event_queue', ListType(tup_type)),
#                        ('control_array', float32[:]),
#                        ('name', unicode_type)]
#
# spec_dynamic_events = OrderedDict(spec_dynamic_events)
# @jitclass(spec_dynamic_events)
# class DynamicEvent(object):
# dynamicEvent is a structure that handles the control over a given array, e.g. the capacities of links
# if a control algorithm should make changes to these
# scheduling of events is handled with a heap data structure
# some of the type choices and casting here are motivated by heapq not accepting heterogeneous type
# control array needs to be a float array

# TODO: think about if there should be DynamicArrayEvents and DynamicCSREvents, can't both be handled by the same class
# dynamically closing a turn is not possible with this design (maybe through capcity?), i guess that is not problematic
# def __init__(self, name, control_array):
#     self.__event_queue = List.empty_list(tup_type)
#     self.__control_array = control_array
#
# def add_event(self, time, obj_index, val):
#     heappush(self.event_queue, (float32(time), float32(obj_index), float32(val)))
#
# def get_next_event(self):
#     (time, obj_index, val) = self.event_queue[0]
#
# def pop_next_event(self):
#     (time, obj_index, val) = heappop(self.event_queue)
# alternative to heap is to have different queues for each dt, continuous should be better for longer time steps..)
