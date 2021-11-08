#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import dyntapy.assignment_context
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.sta.algorithms.deterministic.frank_wolfe import frank_wolfe
from dyntapy.sta.algorithms.deterministic.msa import msa_flow_averaging
from dyntapy.sta.algorithms.deterministic.dial_algorithm_B.bush_manager import dial_b
from dyntapy.sta.algorithms.stochastic.uncongested_dial import uncongested_stochastic_assignment
from dyntapy.sta.algorithms.stochastic.congested_dial import congested_stochastic_assignment
from dyntapy.settings import static_parameters


def DUN(obj:StaticAssignment):
    """

    Returns
    -------

    """
    dyntapy.assignment_context.running_assignment = obj
    raise NotImplementedError
    # TODO: implement this for further exercises ..


def DUE(obj: StaticAssignment, method=None):
    """

    Parameters
    ----------
    method : name of the method to be used, defaults to provided value in config_dict
    g : TrafficGraph
    od_matrix : array like object, dimensions as node x node in TrafficGraph

    Returns
    -------

    """
    dyntapy.assignment_context.running_assignment = obj
    methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
    if method is None:
        method = static_parameters.assignment.methods['DUE']
    if method not in methods:
        raise NotImplementedError
    if method == 'bpr,flow_avg':
        obj.link_travel_times, obj.link_flows = msa_flow_averaging(obj)
    if method == 'frank_wolfe':
        obj.link_travel_times, obj.link_flows = frank_wolfe(obj)
    if method == 'dial_b':
        obj.link_travel_times, obj.link_flows, _ = dial_b(obj)
    return obj.link_flows,obj.link_travel_times


def SUN(obj: StaticAssignment, method=None):
    dyntapy.assignment_context.running_assignment = obj
    methods = ['dial_uncongested']
    if method is None:
        method = static_parameters.assignment.methods['SUN']
    if method not in methods:
        raise NotImplementedError
    if method == 'dial_uncongested':
        obj.link_flows = uncongested_stochastic_assignment(obj)
        obj.link_travel_times = obj.link_ff_times
        print('uncongested dial called')
    return obj.link_flows,obj.link_travel_times


def SUE(obj: StaticAssignment, method=None):
    dyntapy.assignment_context.running_assignment = obj
    raise NotImplementedError
    methods = ['dial_congested']
    if method is None:
        method = static_parameters.assignment.methods['SUE']
    if method not in methods:
        raise NotImplementedError
    if method == 'dial_congested':
        obj.link_flows, costs = congested_stochastic_assignment(obj)
        print('congested dial called')
    return obj.link_flows,obj.link_travel_times
