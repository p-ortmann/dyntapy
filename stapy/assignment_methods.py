#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from stapy.assignment import StaticAssignment
from stapy.algorithms.deterministic.frank_wolfe import frank_wolfe
from stapy.algorithms.deterministic.msa import msa_flow_averaging
from stapy.algorithms.deterministic.dial_algorithm_B.bush_manager import dial_b
from stapy.algorithms.stochastic.uncongested_dial import uncongested_stochastic_assignment
from stapy.settings import assignment_method_defaults


def DUN(g, od_matrix):
    """

    Parameters
    ----------
    g : TrafficGraph
    od_matrix : array like object, dimensions as node x node in TrafficGraph

    Returns
    -------

    """
    obj = StaticAssignment(g, od_matrix)

    obj.write_back(keyed_data=True)
    # TODO: implement this for further exercises ..


def DUE(g, od_matrix, method=None):
    """

    Parameters
    ----------
    method : name of the method to be used, defaults to provided value in config_dict
    g : TrafficGraph
    od_matrix : array like object, dimensions as node x node in TrafficGraph

    Returns
    -------

    """
    methods = ['bpr,flow_avg', 'frank_wolfe', 'dial_b']
    if method is None:
        method = assignment_method_defaults['DUE']
    if method not in methods:
        raise NotImplementedError
    obj = StaticAssignment(g, od_matrix)
    if method == 'bpr,flow_avg':
        obj.link_travel_times, obj.link_flows = msa_flow_averaging(obj)
    if method == 'frank_wolfe':
        obj.link_travel_times, obj.link_flows = frank_wolfe(obj)
    if method == 'dial_b':
        obj.link_travel_times, obj.link_flows, _ = dial_b(obj)
    obj.write_back()
    # add visualization


def SUN(g, od_matrix, method=None):
    methods = ['dial_uncongested']
    if method is None:
        method = assignment_method_defaults['SUN']
    if method not in methods:
        raise NotImplementedError
    obj = StaticAssignment(g, od_matrix)
    if method == 'dial_uncongested':
        obj.link_flows = uncongested_stochastic_assignment(obj)
        obj.link_travel_times = obj.link_ff_times
        print('uncongested dial called')
    obj.write_back()
