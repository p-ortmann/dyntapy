#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numba as nb
from collections import OrderedDict
from dtapy.datastructures.csr import CSRMatrix
from numba.core.types import UniTuple, int64, float64
from numba.core.types.containers import ListType
from numba.typed.typedlist import List
from heapq import heappush, heappop, heapreplace
from dtapy.datastructures.csr import construct_sparse_link_matrix
from numba.experimental import jitclass

spec_link = [('capacity', float64[:]),
             ('from_node', int64[:]),
             ('to_node', int64[:]),
             ('kjam', float64[:]),
             ('length', float64[:]),
             ('flow', float64[:, :, :]),
             ('travel_time', float64[:, :, :])]

spec_link = OrderedDict(spec_link)


@jitclass(spec_link)
class Links(object):
    # link matrix is a float matrix with three dimensions that contains all the information on the link state for all
    # time steps we use transpose here to make access to individual elements easy, e.g. network.links.flows[link_id]
    # returns the flow array for all time steps
    def __init__(self, length, kjam, from_node, to_node, capacity, flow, travel_time):
        self.capacity = capacity
        self.length = length
        self.kjam = kjam
        self.to_node = to_node
        self.from_node = from_node
        self.flow = flow
        self.travel_time = travel_time


spec_node = [('forward', CSRMatrix.class_type.instance_type),
             ('backward', CSRMatrix.class_type.instance_type)]

spec_node = OrderedDict(spec_node)


@jitclass(spec_node)
class Nodes(object):
    # forward and backward are sparse matrices in csr format that indicate connected links and their nodes
    # both are nodes x nodes with f(i,j) = link_id and essentially carry the same information. There's duplication to
    # avoid on-the-fly transformations.
    # forward is fromNode x toNode and backward toNode x fromNode

    def __init__(self, forward: CSRMatrix, backward: CSRMatrix):
        self.forward = forward
        self.backward = backward


@jitclass(spec_turn)
class Turns(object):
    # db_restrictions refer to destination based restrictions as used in recursive logit
    def __init__(self, fractions, db_restrictions: CSRMatrix):
        self.fractions = fractions
        self.db_restrictions = db_restrictions


spec_demand = {'links': Links.class_type.instance_type}


@jitclass(spec_demand)
class Demand(object):
    def __init__(self, od_matrix):
        self.od_matrix = od_matrix
        self.origins


my_instance_type = CSRMatrix.class_type.instance_type
spec_static_event = [('event_csrs', ListType(my_instance_type))]
spec_static_event = OrderedDict(spec_static_event)


@jitclass(spec_static_event)
class StaticEvent(object):
    def __init__(self, name, index_array, values):
        self.csr = construct_sparse_link_matrix(index_array=index_array, values=values)
        self.name = name


tup_type = UniTuple(float64, 3)
spec_dynamic_events = [('event_queue', ListType(tup_type)),
                       ('control_array', float64[:])]

spec_dynamic_events = OrderedDict(spec_dynamic_events)


@jitclass(spec_dynamic_events)
class DynamicEvent(object):
    # dynamicEvent is a structure that handles the control over a given array, e.g. the capacities of links
    # if a control algorithm should make changes to these
    # scheduling of events is handled with a heap data structure
    # some of the type choices and casting here are motivated by heapq not accepting heterogeneous types
    # Network object should carry more than one DynamicEvent if necessary!!!
    # control array needs to be a float array (for now, see create_d

    def __init__(self, control_array, T):
        self.event_queue = List.empty_list(tup_type)
        self.control_array = control_array

    def add_event(self, obj_index, val, time):
        current_val = self.control_array[obj_index]
        heappush(self.event_queue, (float64(time), float64(obj_index), float64(val)))

    def get_next_event(self):
        (time, obj_index, val) = heappop(self.event_queue)


spec_network = [('links', Links.class_type.instance_type),
                ('nodes', Nodes.class_type.instance_type),
                ('turns', Turns.class_type.instance_type),
                ('static_events', ListType(StaticEvent.class_type.instance_type)),
                ('dynamic_events', ListType(DynamicEvent.class_type.instance_type))]


@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.dynamic_events
        self.static_events


def create_dynamic_event_cls(type):
    # if one is interested in creating multiple dynamic events, some of which that may be manipulating an integer array,
    # we'd need a functions that handles the type definition of the specs and returns the appropriate class variant.
    # rebuild the class below and return it ..

    return DynamicEvent
