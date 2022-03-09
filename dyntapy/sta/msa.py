#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from numba import objmode, njit
import numpy as np

from dyntapy.demand import InternalStaticDemand
from dyntapy.settings import parameters
from dyntapy.sta.gap import gap
from dyntapy.sta.utilities import aon, __bpr_cost
from dyntapy.supply import Network
from dyntapy.results import StaticResult, get_skim
from dyntapy._context import iteration_states

gap_definition = "relative gap"
msa_max_iterations = parameters.static_assignment.msa_max_iterations
msa_delta = parameters.static_assignment.msa_delta


@njit
def msa_flow_averaging(
    network: Network, demand: InternalStaticDemand, store_iterations=False
):
    gaps = []
    converged = False
    k = int(0)
    f1 = np.zeros(network.tot_links)
    f2 = f1.copy()
    ff_tt = network.links.length / network.links.free_speed
    while not converged:
        k = k + 1
        if k == 1:
            costs = __bpr_cost(
                capacities=network.links.capacity,
                ff_tts=ff_tt,
                flows=f2,
            )
        else:
            costs = __bpr_cost(
                capacities=network.links.capacity,
                ff_tts=ff_tt,
                flows=f2,
            )
        ssp_costs, f2 = aon(demand, costs, network)
        # print("done")
        if k == 1:
            converged = False
            last_gap = 1
            gaps.append(last_gap)

        else:
            f2 = 1 / k * f2 + (k - 1) / k * f1
            current_gap = gap(f1, costs, demand.to_destinations.values, ssp_costs)
            converged = current_gap < msa_delta or k == msa_max_iterations
            gaps.append(current_gap)
            if current_gap < last_gap:
                best_flow_vector = f1
                best_costs = costs
                last_gap = current_gap
        f1 = f2.copy()
        if store_iterations:
            # lists cannot be passed to object mode
            arr_gap = np.zeros(len(gaps))
            for idx, val in enumerate(gaps):
                arr_gap[idx] = val
            with objmode:
                result = StaticResult(
                    costs,
                    f1,
                    demand.origins,
                    demand.destinations,
                    skim=get_skim(costs, demand, network),
                    gap_definition=gap_definition,
                    gap=np.ndarray(arr_gap),
                )
                iteration_states.append(result)
    return best_costs, best_flow_vector, gap_definition, gaps
