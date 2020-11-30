#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from collections import OrderedDict
from datastructures.csr import CSRMatrix
from numba.core.types import UniTuple, int64, float64, unicode_type
from numba.core.types.containers import ListType
from numba.typed.typedlist import List
from heapq import heappush, heappop
from datastructures.csr import construct_sparse_link_matrix
from numba.experimental import jitclass
from dtapy.parameters import link_capacity_id, registered_events

csr_type = CSRMatrix.class_type.instance_type

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
    # Links carries all the attribute arrays like capacity, kjam etc and also forward and backward - CSR matrices that
    # indicate connected turns
    def __init__(self, length, kjam, from_node, to_node, capacity, flow, travel_time, sending_flow, receiving_flow):
        self.capacity = capacity
        self.length = length
        self.kjam = kjam
        self.to_node = to_node
        self.from_node = from_node
        self.flow = flow
        self.travel_time = travel_time
        self.sending_flow = sending_flow  # accessed for multiple time periods or a single one?
        self.receiving_flow = receiving_flow
        self.forward  # csr linkxlink row is outgoing turns
        self.backward  # csr incoming turns
        self.event_changes  # csr link_id x attribute_ids = val, checks necessary if future values are extracted

    if registered_events[link_capacity_id]:
        def get_capacity(self, link_id, time=0):  # timing access and overhead through the sparse check,
            # possibly different approach, depends on whether full array is needed for the 'time'
            # or only a single entry
            # alternative: add a check on whether event queue exists for this attribute (possibly hard, at compile time the object
            # doesn't exist yet ...# - can functions be compiled differently based on this?
            # TODO: test this design on compatibility with numba
            if time == 0:
                return self.capacity[link_id]
            else:
                if self.event_changes.get_nnz(link_id) != link_capacity_id:
                    return self.capacity[link_id]
                else:
                    # interpolation ..
                    return 0
    else:
        def get_capacity(self, link_id, time=0): # this will get inlined by the jit compiler - no overhead .
            return self.capacity[link_id]
    def get_var(self, link_id, attribute, time=0):
        pass

    # TODO: explore and enforce use of @property decorator for those properties that are allowed public access
    # TODO: name mangle properties that should remain private


spec_results = [('',), ]


@jitclass(spec_results)
class Results(object):
    def __init__(self, turning_fractions, flows):
        self.turning_fractions
        self.flows
        self.path_set  # list of all used paths by od pair
        self.controller_strategy


spec_node = [('forward', csr_type),
             ('backward', csr_type)]

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
        self.position_first_in
        self.position_first_out
        self.nbtf
        self.event_changes
        #nodes.forward.getrow(node_id)=link_ids


spec_turn = [('fractions', csr_type)]


@jitclass(spec_turn)
class Turns(object):
    # db_restrictions refer to destination based restrictions as used in recursive logit
    def __init__(self, fractions, db_restrictions: CSRMatrix, receiving_flow):
        self.fractions = fractions  # node x turn_ids
        self.db_restrictions = db_restrictions  # destinations x turns,
        self.receiving_flow = receiving_flow
        self.event_changes


spec_demand = {'links': Links.class_type.instance_type}


@jitclass(spec_demand)
class Demand(object):
    def __init__(self, od_matrix, origins, destinations):
        self.od_matrix = od_matrix  # csr matrix origins x destinations #maybe also implement inverse ..
        self.origins = origins  # array of node id's that are origings
        self.destinations = destinations  # array of node id's destinations


spec_static_event = [('events', csr_type),
                     ('attribute_id', int64)]
spec_static_event = OrderedDict(spec_static_event)


@jitclass(spec_static_event)
class StaticEvent(object):
    def __init__(self, attribute_id, index_array, values):
        self.events = construct_sparse_link_matrix(index_array=index_array, values=values)
        self.attribute_id = attribute_id


tup_type = UniTuple(float64, 3)
spec_dynamic_events = [('event_queue', ListType(tup_type)),
                       ('control_array', float64[:]),
                       ('name', unicode_type)]

spec_dynamic_events = OrderedDict(spec_dynamic_events)


@jitclass(spec_dynamic_events)
class DynamicEvent(object):
    # dynamicEvent is a structure that handles the control over a given array, e.g. the capacities of links
    # if a control algorithm should make changes to these
    # scheduling of events is handled with a heap data structure
    # some of the type choices and casting here are motivated by heapq not accepting heterogeneous types
    # Network object should carry more than one DynamicEvent if necessary!!!
    # control array needs to be a float array (for now, see create_d

    # TODO: think about if there should be DynamicArrayEvents and DynamicCSREvents, can't both be handled by the same class
    # dynamically closing a turn is not possible with this design (maybe through capcity?), i guess that is not problematic
    def __init__(self, name, control_array):
        self.__event_queue = List.empty_list(tup_type)
        self.__control_array = control_array

    def add_event(self, time, obj_index, val):
        heappush(self.event_queue, (float64(time), float64(obj_index), float64(val)))

    def get_next_event(self):
        (time, obj_index, val) = self.event_queue[0]

    def pop_next_event(self):
        (time, obj_index, val) = heappop(self.event_queue)
    # alternative to heap is to have different queues for each dt, continuous should be better for longer time steps..)


spec_network = [('links', Links.class_type.instance_type),
                ('nodes', Nodes.class_type.instance_type),
                ('turns', Turns.class_type.instance_type),
                ('static_events', ListType(StaticEvent.class_type.instance_type)),
                ('dynamic_events', ListType(DynamicEvent.class_type.instance_type)),
                ('tot_links', int64),
                ('tot_nodes', int64),
                ('tot_destinations', int64)]


@jitclass(spec_network)
class Network(object):
    # link mat
    def __init__(self, links, nodes, turns, dynamic_events, static_events, tot_links, tot_nodes, tot_destinations):
        self.links = links
        self.nodes = nodes
        self.turns = turns
        self.dynamic_events = dynamic_events
        self.static_events = static_events
        self.tot_links = tot_links
        self.tot_nodes = tot_nodes
        self.tot_destinations = tot_destinations
        self.initial_state_based_on_control
        # TODO: add lookup tables for name to index

    def update_event_changes(self, dt, T):
        # dt current time step
        # T time horizon
        # sets event changes in links turns nodes in accordance with what is registered in dynamic events and
        # static events for that time period
        # reduces overhead induced by these event mechanisms ..
        # restricts controller to register changes in time horizon steps, is this
        pass
    def get_controller_response(self,dt,T, Results:Results):
        pass



def create_dynamic_event_cls(type):
    # if one is interested in creating multiple dynamic events, some of which that may be manipulating an integer array,
    # we'd need a functions that handles the type definition of the specs and returns the appropriate class variant.
    # rebuild the class below and return it ..

    return DynamicEvent

# how to handle dynamic changes, proposal: 3 column array with time stamp and value, temporary structutre for computations
# that reach into the past with +- k * dt(travel time on longest link), rewritten for each dt in the simulation
# alternative:
# register sparse matrix for each object, with obj_id x attribute_id=val as change in analysis TP, retrieval for specific time has
# overhead of checking the sparse structure ..
