#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from collections import OrderedDict
from datastructures.csr import UI32CSRMatrix, F32CSRMatrix
from numba.core.types import float32, uint32, int8
from numba.core.types.containers import ListType
from numba.experimental import jitclass

ui32csr_type = UI32CSRMatrix.class_type.instance_type
f32csr_type = F32CSRMatrix.class_type.instance_type

# possibly: experiment with data types uint32, float32
# -> int32(max 2147483648), float32( worst precision under 16384 is 0.001953125)

spec_link = [('capacity', float32[:]),
             ('from_node', uint32[:]),
             ('to_node', uint32[:]),
             ('length', float32[:]),
             ('flow', float32[:, :]),
             ('travel_time', float32[:, :]),
             ('forward', ui32csr_type),
             ('backward', ui32csr_type)]

spec_link = OrderedDict(spec_link)


@jitclass(spec_link)
class Links(object):
    """
    A simple class that carries various arrays and CSR Matrices that have link ids as their index
    """

    def __init__(self, length, from_node, to_node, capacity, v_wave, costs, v0, cvn_up, cvn_down,
                 forward, backward):
        self.capacity = capacity
        self.length = length
        self.to_node = to_node
        self.from_node = from_node
        self.v_wave = v_wave
        self.costs = costs
        self.v0 = v0
        self.cvn_up = cvn_up
        self.cvn_down = cvn_down
        self.forward = forward  # csr linkxlink row is outgoing turns
        self.backward = backward  # csr incoming turns


spec_results = [('',), ]


@jitclass(spec_results)
class Results(object):
    def __init__(self, turning_fractions, flows):
        self.turning_fractions
        self.flows
        self.path_set  # list of all used paths by od pair
        self.controller_strategy


spec_node = [('forward', ui32csr_type),
             ('backward', ui32csr_type),
             ('control_type', int8[:]),
             ('capacity', float32[:])]

spec_node = OrderedDict(spec_node)


@jitclass(spec_node)
class Nodes(object):
    """
    A simple class that carries various arrays and CSR Matrices that have node ids as their index
    """

    def __init__(self, forward: UI32CSRMatrix, backward: UI32CSRMatrix, control_type, capacity):
        """
        forward and backward are sparse matrices in csr format that indicate connected links and their nodes
        both are nodes x nodes with f(i,j) = link_id and essentially carry the same information. There's duplication to
        avoid on-the-fly transformations.
        forward is fromNode x toNode and backward toNode x fromNode
        Parameters
        ----------
        forward : I64CSRMatrix <uint32>
        backward : I64CSRMatrix <uint32>
        turn_fractions : F64CSRMatrix <float32>

        """
        self.forward : UI32CSRMatrix = forward
        self.backward = backward
        #self.turn_fractions = turn_fractions  # node x turn_ids
        self.control_type = control_type  #
        self.capacity = capacity


spec_turn = [('db_restrictions', ui32csr_type),
             ('t0', float32),
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
    def __init__(self, db_restrictions: UI32CSRMatrix, t0, capacity, from_node, via_node, to_node, from_link, to_link,
                 type):
        """

        Parameters
        ----------
        db_restrictions :
        """
        self.db_restrictions = db_restrictions  # destinations x turns
        self.t0 = t0
        self.capacity = capacity
        self.from_node = from_node
        self.to_node = to_node
        self.via_node = via_node
        self.from_link = from_link
        self.to_link = to_link
        self.type = type


spec_demand = ['to_destinations', ui32csr_type,
               'to_origins', f32csr_type]
spec_demand = OrderedDict(spec_demand)


@jitclass(spec_demand)
class Demand(object):
    def __init__(self, od_matrix, origins, destinations, number_of_time_steps):
        self.od_matrix = od_matrix  # csr matrix origins x destinations #maybe also implement inverse ..
        self.origins = origins  # array of node id's that are origins
        self.destinations = destinations  # array of node id's destinations
        self.number_of_timesteps = number_of_time_steps


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
                ('tot_destinations', uint32)]


@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns, static_events, tot_links, tot_nodes, tot_destinations):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.static_events = static_events
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_destinations = tot_destinations
        # TODO: add lookup tables for name to index

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
