#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
from stapy.algorithms.helper_funcs import calculate_costs, aon
from stapy.settings import assignment_parameters
from stapy.assignment import StaticAssignment
from stapy.gap import gap
from utilities import log


def msa_flow_averaging(obj: StaticAssignment):
    """

    Parameters
    ----------
    obj : StaticAssignment object

    Returns
    -------

    """
    msa_max_iterations = assignment_parameters['msa_max_iterations']
    msa_delta = assignment_parameters['msa_delta']
    number_of_od_pairs = len(obj.od_flow_vector)
    converged = False
    k = np.int(0)
    f1 = obj.link_flows  # initializing flow list with 0
    f2 = f1.copy()
    while not converged:
        k = k + 1
        if k == 1:
            costs = obj.link_ff_times
            costs = calculate_costs(link_capacities=obj.link_capacities, link_ff_times=obj.link_ff_times, link_flows=f2)
            log('starting msa')
        else:
            costs = calculate_costs(link_capacities=obj.link_capacities, link_ff_times=obj.link_ff_times, link_flows=f2)
        #print('aon run')
        ssp_costs,f2 = aon(obj.demand_dict,costs, obj.forward_star, obj.edge_map, number_of_od_pairs, obj.node_order)
        #print("done")
        if k==1:
            converged=False
            log('first aon run')
            last_gap=1
        else:
            f2 = 1 / k*f2+(k-1)/k*f1
            current_gap = gap(f1, costs, obj.od_flow_vector, ssp_costs)
            converged = current_gap < msa_delta or k == msa_max_iterations
            if current_gap<last_gap:
                best_flow_vector=f1
                best_costs=costs
                last_gap=current_gap
            if not converged:
                log(f'current gap is{current_gap}')
        f1=f2.copy()
        if k>1:
            obj.store_iteration(flow_vector=f1, gap=current_gap)
    return best_costs, best_flow_vector

