#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
from dyntapy.sta.algorithms.helper_funcs import calculate_costs, aon
from dyntapy.settings import  static_parameters
from dyntapy.sta.assignment import StaticAssignment
from dyntapy.sta.gap import gap
from dyntapy.utilities import log
from dyntapy.visualization import show_network
def msa_flow_averaging(obj: StaticAssignment):
    """

    Parameters
    ----------
    obj : StaticAssignment object

    Returns
    -------

    """
    msa_max_iterations =static_parameters.assignment.msa_max_iterations
    msa_delta =static_parameters.assignment.msa_delta
    number_of_od_pairs = obj.demand.to_destinations.values.size
    converged = False
    k = np.int(0)
    f1 = obj.link_flows  # initializing flow list with 0
    f2 = f1.copy()
    while not converged:
        k = k + 1
        if k == 1:
            costs = calculate_costs(link_capacities=obj.link_capacities, link_ff_times=obj.link_ff_times, link_flows=f2)
            log('starting msa')
        else:
            costs = calculate_costs(link_capacities=obj.link_capacities, link_ff_times=obj.link_ff_times, link_flows=f2)
        ssp_costs,f2 = aon(obj.demand, costs, obj.out_links, obj.edge_map, number_of_od_pairs, obj.tot_nodes)
        #print("done")
        if k==1:
            converged=False
            log('first aon run', to_console=True)
            last_gap=1
        else:
            f2 = 1 / k*f2+(k-1)/k*f1
            current_gap = gap(f1, costs, obj.demand.to_destinations.values,ssp_costs)
            converged = current_gap < msa_delta or k == msa_max_iterations
            if current_gap<last_gap:
                best_flow_vector=f1
                best_costs=costs
                last_gap=current_gap
            if not converged:
                log(f'current gap is{current_gap}', to_console=True)
        f1=f2.copy()
    return best_costs, best_flow_vector

