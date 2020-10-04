#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# the
from stapy.assignment import StaticAssignment
from stapy.settings import assignment_parameters
from numba import njit, jit
from numba.typed import Dict, List
from stapy.algorithms.graph_utils import __shortest_path, __pred_to_epath2, make_forward_stars, make_backward_stars
from stapy.algorithms.helper_funcs import __valid_edges, __topological_order, __bpr_cost
from stapy.settings import assignment_parameters
import numpy as np
from stapy.algorithms.stochastic.uncongested_dial import load_all_bushes, generate_bushes


def congested_stochastic_assignment(obj: StaticAssignment):
    theta = assignment_parameters['logit_theta']
    msa_max_iterations = assignment_parameters['msa_max_iterations']
    msa_delta = assignment_parameters['msa_delta']
    # print(f'my theta is {theta}')
    topological_orders, edges, L, largest_destination_labels = generate_bushes(obj.link_ff_times, obj.edge_map,
                                                                               obj.forward_star,
                                                                               obj.demand_dict, obj.node_order)
    new_flows = load_all_bushes(obj, topological_orders, edges, L, largest_destination_labels, obj.link_ff_times, theta)
    gap = 1
    k = 1
    multiphase_counter = 0
    while gap < assignment_parameters['SUE_gap']:
        k = k + 1
        costs = __bpr_cost(new_flows, obj.link_capacities, obj.link_ff_times)
        topological_orders, edges, L, largest_destination_labels = generate_bushes(costs, obj.edge_map,
                                                                                   obj.forward_star,
                                                                                   obj.demand_dict, obj.node_order)
        old_flows = new_flows.copy()
        new_flows = load_all_bushes(obj, topological_orders, edges, L, largest_destination_labels, costs, theta)
        new_flows = 1 / k * new_flows + (k - 1) / k * old_flows
        converged = np.sum(np.abs(new_flows - old_flows)) / (
                    np.sum(new_flows + old_flows) / 2) < msa_delta or k == msa_max_iterations
        multiphase_counter += 1
        if converged:
            break
        if multiphase_counter == 5:
            k = 1
    return new_flows
