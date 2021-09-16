#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import math

import numpy as np

from dyntapy.sta.algorithms.helper_funcs import calculate_costs, aon
from dyntapy.sta.assignment import StaticAssignment as __StaticAssignment
from dyntapy.settings import  static_parameters
from dyntapy.utilities import log


def frank_wolfe(obj: __StaticAssignment):
    assert type(obj) is __StaticAssignment
    max_iterations = static_parameters.assignment.fw_max_iterations
    delta =static_parameters.assignment.fw_delta
    converged = False
    f1, f2 = np.zeros(obj.tot_links), np.zeros(obj.tot_links)  # initializing flow list with 0
    state = np.array(list(zip(f2, obj.link_capacities, obj.link_ff_times)))
    counter = int(0)
    c1, c2 = np.double(0), np.double(1)
    log('Frank - Wolfe initiated', 20)
    while not converged:
        counter += 1
        f1 = f2
        costs = calculate_costs(link_capacities=obj.link_capacities, link_ff_times=obj.link_ff_times, link_flows=f2)
        _,f2 = aon(obj.demand, costs, obj.out_links, obj.edge_map, obj.demand.to_destinations.values.size, obj.tot_nodes)
        if counter > 1:
            f2, c2 = __gssrec(__eval_obj_func_fw, f1, f2, state)
            log(f'iteration costs c2 {c2} and c1 {c1}', 10)
        converged = abs(c2 - c1) / c2 < delta or counter == max_iterations
        c1 = c2.copy()
    log(f'convergence reached in {counter} iterations', 20)
    return costs,f2


def __eval_obj_func_fw(state):
    bpr_beta = static_parameters.assignment.bpr_beta
    bpr_alpha =static_parameters.assignment.bpr_alpha

    def integral(state_array):
        # state[2] - free flow travel times, state[1] - capacities, state[0] - flows
        # calculates integral of bpr function for all links
        integral_value = state_array[2] * (
                (bpr_alpha * state_array[0] * (state_array[0] / state_array[1]) ** bpr_beta) /
                (bpr_beta + 1) + state_array[0])
        return integral_value

    objective_value = np.sum(np.apply_along_axis(integral, 1, state))
    return objective_value


invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def __gssrec(f, fa, fb, state, tol=1e-3, delta_f=None, fc=None, fd=None, func_c=None, func_d=None):
    """ Golden section search, recursive.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    """
    state[:, 0] = fa
    state[:, 0] = fb
    if delta_f is None:
        delta_f = fb - fa
    if fc is None:
        fc = fa + invphi2 * delta_f
    if fd is None:
        fd = fa + invphi * delta_f
    if func_c is None:
        state[:, 0] = fc
        func_c = f(state)
    if func_d is None:
        state[:, 0] = fd
        func_d = f(state)
    if abs(func_c - func_d) / func_c <= tol:
        return fa, max(func_c, func_d)
    if func_c < func_d:
        return __gssrec(f, fa, fb, state, tol, delta_f=None, fc=None, func_c=None, fd=fd, func_d=func_c)
    else:
        return __gssrec(f, fa, fb, state, tol, delta_f=None, fc=fc, func_c=func_d, fd=None, func_d=None)
