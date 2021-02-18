#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dtapy.core.network_loading.link_models.i_ltm_cls import ILTMState
from dtapy.core.network_objects_cls import SimulationTime, Network
from dtapy.assignment import Assignment
from dtapy.parameters import route_choice_dt, route_choice_agg
from dtapy.core.route_choice.dynamic_dijkstra import dijkstra
from dtapy.core.route_choice.aon_cls import AONState
import numpy as np
from dtapy.parameters import route_choice_delta
from heapq import heappop, heappush


# TODO: add generic results object
@njit
def update_arrival_maps(assignment: Assignment, state:AONState):

    tot_time_steps=assignment.time.tot_time_steps
    from_node = assignment.network.links.from_node
    out_links =  assignment.network.nodes.out_links
    all_destinations = assignment.dynamic_demand.all_destinations
    state.prev_costs = state.cur_costs
    state.cur_costs = assignment.results.costs
    delta_costs =  np.abs(state.cur_costs-state.prev_costs)
    nodes_2_update = [(np.float32(0), np.float32(0))]
    nodes_2_update.pop(0) # list has to be initialized for type inference before heappush ..
    arrival_maps= state.arrival_maps


    for destination in all_destinations:
        for t in range(tot_time_steps,0, -1):
                for link, delta in np.ndenumerate(delta_costs[t,:]):
                    # find all links with changed travel times and add their tail nodes
                    # to the heap of nodes to be updated
                    if delta> route_choice_delta:
                        node = from_node[link]
                        dist = arrival_maps[destination, t, node]
                        heappush(nodes_2_update,(dist, np.float32(node)))
                        nodes_2_update.append(from_node[link])
                while len(nodes_2_update)>0:
                    heap_item  = heappop(nodes_2_update)
                    node= np.uint32(heap_item[1])
                    new_dist =  np.inf
                    for link in out_links.get_nnz(node):
                        pass














