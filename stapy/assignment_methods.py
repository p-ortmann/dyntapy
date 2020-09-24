#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from stapy.assignment import StaticAssignment
from stapy.algorithms.frank_wolfe import frank_wolfe
from stapy.algorithms.msa import msa_flow_averaging
from stapy.algorithms.dial_algorithm_B.bush_manager import dial_b
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
        obj.link_travel_times,obj.link_flows = frank_wolfe(obj)
    if method =='dial_b':
        obj.link_travel_times, obj.link_flows, _ = dial_b(obj)
    obj.write_back()
    # add visualization


def SUE(g, od_matrix, method=None):
    methods = ['dial']
    if method is None:
        method = assignment_method_defaults['SUE']
    if method not in methods:
        raise NotImplementedError
    obj = StaticAssignment(g, od_matrix)
    if method == 'dial':
        obj.link_flows = dial_b(obj)
        print('dial called')
    obj.write_back()
