#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from collections import OrderedDict
from datastructures.csr import UI32CSRMatrix, F32CSRMatrix, UI8CSRMatrix
from numba.core.types import float32, uint32, int32, int8, uint8, boolean
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from numba import njit
import numpy as np

ui32csr_type = UI32CSRMatrix.class_type.instance_type
f32csr_type = F32CSRMatrix.class_type.instance_type
ui8csr_type = UI8CSRMatrix.class_type.instance_type

# We differentiate here between the generic Links Nodes and Turns object and more specialized objects that inherit
# from these classes Dynamic Traffic Assignment algorithms all use a base line of attributes that are kept in these
# baseline classes Algorithms like LTM need additional attributes - these are kept in objects that inherit from their
# respective class, so LTM has a special class for its links to store things like cvn .. the source code for the
# baseline classes is replicated both in jitclass decorated and undecorated (uncompiled) form because inheritance
# otherwise does not work at this point, see https://github.com/numba/numba/issues/1694 .
# adding a new assignment algorithm to this is meant to be made simpler this way, nothing changes in the assignment object
# one simply needs to define new extending network, turn, node and link objects as needed and write a setup file, as shown for i-ltm.

spec_link = [('capacity', float32[:]),
             ('from_node', uint32[:]),
             ('to_node', uint32[:]),
             ('length', float32[:]),
             ('flow', float32[:, :]),
             ('costs', float32[:, :]),
             ('out_turns', ui32csr_type),
             ('in_turns', ui32csr_type),
             ('v_wave', float32[:]),
             ('v0', float32[:]),
             ('type', int8[:]),
             ('lanes', int8[:])]


# spec_link = OrderedDict(spec_link)
@jitclass(spec_link)
class Links(object):
    """
    A simple class that carries various arrays and CSR Matrices that have link ids as their index
    """

    def __init__(self, length, from_node, to_node, capacity, v_wave, costs, v0,
                 out_turns, in_turns, lanes):
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.v_wave = v_wave
        self.costs = costs
        self.v0 = v0
        self.out_turns = out_turns  # csr linkxlink row is outgoing turns
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes


class UncompiledLinks(object):
    """
    A simple class that carries various arrays and CSR Matrices that have link ids as their index
    """

    def __init__(self, length, from_node, to_node, capacity, v_wave, costs, v0,
                 out_turns, in_turns, lanes):
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.v_wave = v_wave
        self.costs = costs
        self.v0 = v0
        self.out_turns = out_turns  # csr linkxlink row is outgoing turns
        self.in_turns = in_turns  # csr incoming turns
        self.lanes = lanes


spec_iltm_link = [('links', Links.class_type.instance_type),
                  ('kjam', float32[:]),
                  ('kcrit', float32[:]),
                  ('cvn_up', float32[:, :]),
                  ('cvn_down', float32[:, :]),
                  ('vf_index', int32[:]),
                  ('vf_ratio', float32[:]),
                  ('vw_index', int32[:]),
                  ('vw_ratio', float32[:])]


# spec_iltm_link = OrderedDict(spec_iltm_link)


@jitclass(spec_link + spec_iltm_link)
class ILTMLinks(UncompiledLinks):
    __init__Links = UncompiledLinks.__init__

    def __init__(self, links, cvn_up, cvn_down, vf_index, vw_index, vf_ratio, vw_ratio):
        self.__init__Links(links.length, links.from_node, links.to_node, links.capacity, links.v_wave, links.costs,
                           links.v0, links.out_turns, links.in_turns, links.lanes)
        self.cvn_up = cvn_up
        self.cvn_down = cvn_down
        self.vf_index = vf_index
        self.vw_index = vw_index
        self.vf_ratio = vf_ratio
        self.vw_ratio = vw_ratio


# spec_results = [('',), ]
#
#
# @jitclass(spec_results)
# class Results(object):
#     def __init__(self, turning_fractions, flows):
#         self.turning_fractions
#         self.flows
#         self.path_set  # list of all used paths by od pair
#         self.controller_strategy


spec_node = [('out_links', ui32csr_type),
             ('in_links', ui32csr_type),
             ('control_type', int8[:]),
             ('capacity', float32[:]),
             ('tot_out_links', uint32[:]),
             ('tot_in_links', uint32[:])]


# spec_node = OrderedDict(spec_node)


@jitclass(spec_node)
class Nodes(object):
    """
    A simple class that carries various arrays and CSR Matrices that have node ids as their index
    """

    def __init__(self, out_links: UI32CSRMatrix, in_links: UI32CSRMatrix, tot_out_links, tot_in_links, control_type,
                 capacity):
        """
        out_links and in_links are sparse matrices in csr format that indicate connected links and their nodes
        both are nodes x nodes with f(i,j) = link_id and essentially carry the same information. There's duplication to
        avoid on-the-fly transformations.
        out_links is fromNode x Link and in_links toNode x Link in dim with toNode and fromNode as val, respectively.
        Parameters
        ----------
        out_links : I64CSRMatrix <uint32>
        in_links : I64CSRMatrix <uint32>
        """
        self.out_links: UI32CSRMatrix = out_links
        self.in_links = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        # self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity


class UncompiledNodes(object):
    """
    A simple class that carries various arrays and CSR Matrices that have node ids as their index
    """

    def __init__(self, out_links: UI32CSRMatrix, in_links: UI32CSRMatrix, tot_out_links, tot_in_links, control_type,
                 capacity):
        """
        forward and backward are sparse matrices in csr format that indicate connected links and their nodes
        both are nodes x nodes with f(i,j) = link_id and essentially carry the same information. There's duplication to
        avoid on-the-fly transformations.
        forward is fromNode x toNode and backward toNode x fromNode
        Parameters
        ----------
        out_links : I64CSRMatrix <uint32>
        in_links : I64CSRMatrix <uint32>
        turn_fractions : F64CSRMatrix <float32>

        """
        self.out_links: UI32CSRMatrix = out_links
        self.in_links = in_links
        self.tot_out_links = tot_out_links
        self.tot_in_links = tot_in_links
        # self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity


spec_iltm_node = [('nodes', Nodes.class_type.instance_type),
                  ('turn_based_in_links', ui8csr_type),
                  ('turn_based_out_links', ui8csr_type)
                  ]


@jitclass(spec_node + spec_iltm_node)
class ILTMNodes(UncompiledNodes):
    __init__Nodes = UncompiledNodes.__init__

    def __init__(self, nodes, turn_based_in_links, turn_based_out_links):
        """

        Parameters
        ----------
        nodes : Nodes.class_type.instance_type, baseline node object
        turn_based_in_links : csr matrix node x turns
        turn_based_out_links : csr matrix node x turns

        the values of the turn_based - in and out_link csr matrices are the index
        of the corresponding sending and receiving flow vector that the node model receives, capacities are also given
        ordered by in- and out links, see technical.md.
        """
        self.__init__Nodes(nodes.out_links, nodes.in_links, nodes.tot_out_links, nodes.tot_in_links, nodes.control_type,
                           nodes.capacity)

        self.turn_based_in_links = turn_based_in_links
        self.turn_based_out_links = turn_based_out_links


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


spec_time = [('start', float32),
             ('end', float32),
             ('step_size', float32),
             ('tot_time_steps', uint32)]


@jitclass(spec_time)
class SimulationTime(object):
    def __init__(self, start, end, step_size):
        self.start = start
        self.end = end
        self.step_size = step_size
        self.tot_time_steps = uint32(len(np.arange(start, end, step_size)))


spec_demand = [('to_destinations', f32csr_type),
               ('to_origins', f32csr_type),
               ('origins', uint32[:]),
               ('destinations', uint32[:]),
               ('time_step', uint32)]
spec_demand = OrderedDict(spec_demand)


@jitclass(spec_demand)
class StaticDemand(object):
    def __init__(self, to_destinations, to_origins, origins, destinations, time_step):
        self.to_destinations = to_destinations  # csr matrix origins x destinations
        self.to_origins = to_origins  # csr destinations x origins
        self.origins = origins  # array of node id's that are origins
        self.destinations = destinations  # array of node id's destinations
        self.time_step = time_step  # time at which this demand is added to the network


spec_simulation = [('next', StaticDemand.class_type.instance_type),
                   ('demands', ListType(StaticDemand.class_type.instance_type)),
                   ('is_loading', boolean),
                   ('__time_step', uint32),
                   ('all_destinations', uint32[:]),
                   ('all_origins', uint32[:]),
                   ('tot_time_steps', uint32)]


@jitclass(spec_simulation)
class DemandSimulation(object):
    def __init__(self, demands, tot_time_steps):
        self.demands = demands
        self.next = demands[0]
        self.__time_step = uint32(0)  # current simulation time in time step reference
        self.is_loading = self.next.time_step == self.__time_step  # boolean that indicates if during the current
        # time step traffic is loaded into the network
        self.all_destinations = get_all_destinations(demands)  # for destination/origin based labels
        self.all_origins = get_all_origins(demands)
        self.tot_time_steps = tot_time_steps

    def update(self):
        self.__time_step += uint32(1)
        if self.__time_step > self.tot_time_steps:
            raise AssertionError('exceeding limit of time steps as defined in simulation time, use reset')
        if self.__time_step > self.next.time_step:
            self.next = self.demands[self.__time_step]
        if self.next.time_step == self.__time_step:
            self.is_loading = True
        else:
            self.is_loading = False

    def reset(self):
        self.__time_step = uint32(0)
        self.next = self.demands[0]


@njit
def get_all_destinations(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].destinations
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: StaticDemand
        current = np.concatenate((demand.destinations, previous))
        previous = current
    return np.unique(current)


@njit
def get_all_origins(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].origins
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: StaticDemand
        current = np.concatenate((demand.origins, previous))
        previous = current
    return np.unique(current)


spec_static_event = [('events', f32csr_type),
                     ('attribute_id', uint32)]
spec_static_event = OrderedDict(spec_static_event)


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


spec_network = [('links', Links.class_type.instance_type),
                ('nodes', Nodes.class_type.instance_type),
                ('turns', Turns.class_type.instance_type),
                ('static_events', ListType(StaticEvent.class_type.instance_type)),
                ('tot_links', uint32),
                ('tot_nodes', uint32),
                ('tot_turns', uint32)]


@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
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
spec_uncompiled_network = [
    ('static_events', ListType(StaticEvent.class_type.instance_type)),
    ('tot_links', uint32),
    ('tot_nodes', uint32),
    ('tot_turns', uint32)]


class UncompiledNetwork(object):
    # link mat
    def __init__(self, links, nodes, turns, tot_links, tot_nodes, tot_turns):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_turns = tot_turns
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


spec_iltm_network = [('network', Network.class_type.instance_type),
                     ('links', ILTMLinks.class_type.instance_type),
                     ('nodes', ILTMNodes.class_type.instance_type),
                     ('turns', Turns.class_type.instance_type)]


@jitclass(spec_uncompiled_network + spec_iltm_network)
class ILTMNetwork(UncompiledNetwork):
    __init__Network = UncompiledNetwork.__init__

    def __init__(self, network, links, nodes, turns):
        self.__init__Network(links, nodes, turns, network.tot_links, network.tot_nodes, network.tot_turns)

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
