#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from heapq import heappop, heappush
import numpy as np
from numba import njit
from numba.typed import Dict, List


@njit
def make_out_links(adj_list, number_of_nodes):
    forward_star = Dict()
    star_sizes = np.zeros(number_of_nodes, dtype=np.int64)
    nodes = np.arange(number_of_nodes, dtype=np.int64)
    for i in nodes:
        forward_star[i] = np.empty(20, dtype=np.int64)  # In traffic networks nodes will never have more than 10 outgoing edges..
    for edge in adj_list:
        i = edge[0]
        j = edge[1]
        forward_star[i][star_sizes[i]] = j
        star_sizes[i] += 1
    for i in nodes:
        forward_star[i] = forward_star[i][:star_sizes[i]]
    return forward_star


@njit
def make_in_links(adj_list, number_of_nodes):
    backward_star = Dict()
    star_sizes = np.zeros(number_of_nodes, dtype=np.int64) #, dtype=int_dtype
    nodes = np.arange(number_of_nodes, dtype=np.int64)
    for i in nodes:
        backward_star[i] = np.empty(20, dtype=np.int64)  # nodes in traffic networks have less than 10 outgoing edges..
    for edge in adj_list:
        i = edge[0]
        j = edge[1]
        #print(f'i: {i} j: {j} ')
        backward_star[j][star_sizes[j]] = i
        star_sizes[j] += 1
    for i in nodes:
        backward_star[i] = backward_star[i][:star_sizes[i]]
    return backward_star

def node_to_edge_path(node_path, edge_map):
    edge_path=List()
    for id,node in enumerate(node_path[:-1]):
        edge_path.append(edge_map[(node, node_path[id+1])])
    return edge_path



def get_shortest_paths(costs, forward_star, edge_map, source, targets=None, output='epath'):
    """

    Parameters
    ----------
    costs :
    forward_star :
    edge_map :
    source :
    targets :
    output :

    Returns
    -------

    """
    if not targets:
        targets=np.empty(0) # instead of None check inside of jitted shortest path - numba cannot deal with branching on types
    node_order = len(forward_star)
    source = int(source)
    dist, pred = __shortest_path(costs, forward_star, edge_map, source, targets, node_order)
    if output == 'epath':
        return __pred_to_epath(pred, source, targets, edge_map, costs)
    else:
        return dist, pred


@njit
def __shortest_path(costs, forward_star, edge_map, source, targets, node_order):
    """
    typical dijkstra implementation with heaps
    Parameters
    ----------
    costs : float64 vector
    forward_star :numba typed Dict
    edge_map : numba typed Dict (i,j) -> link_id
    source :
    targets :
    node_order :

    Returns
    -------

    """
    #print (costs)
    dist = Dict()  # dictionary of final distances
    seen = Dict()
    pred = Dict()
    my_heap = []
    assert 0 <= source < node_order
    seen[source] = float(0)
    heap_item = (float(0), float(source))
    my_heap.append(heap_item)
    while my_heap:
        heap_item = heappop(my_heap)
        d = heap_item[0]
        i = np.int64(heap_item[1])
        if i in dist:
            continue  # had this node already
        dist[i] = d
        if len(targets) == 0:  # instead of None check - numba cannot deal with branching on types
            pass
        else:
            for index, target in enumerate(targets):
                if i == target:
                    targets = np.delete(targets, index)
                    break
            if len(targets) == 0:
                break
        for j in forward_star[i]:
            ij_dist = dist[i] + costs[edge_map[(i, j)]]
            if j not in seen or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (ij_dist, float(j))
                heappush(my_heap, heap_item)
                pred[j] = i
    assert len(targets) == 0
    # otherwise the provided network is not a strongly connected component ..
    return dist, pred


@njit(nogil=True)
def __pred_to_epath(pred, source, targets, edge_map, costs):
    for target in targets:
        assert target != source
    path_costs = List()
    edge_paths = List()
    for j in targets:
        cost = 0
        path = List()
        path.append(edge_map[(pred[j], j)])
        cost += costs[edge_map[(pred[j], j)]]
        j = pred[j]
        while j != source:
            i = pred[j]
            path.append(edge_map[(i, j)])
            cost += costs[edge_map[(i, j)]]
            j = pred[j]
        edge_paths.append(path)
        path_costs.append(cost)

    return path_costs, edge_paths

@njit(nogil=True)
def __pred_to_epath2(pred, source, targets, edge_map):
    for target in targets:
        assert target != source
    edge_paths = List()
    for j in targets:
        path = List()
        path.append(edge_map[(pred[j], j)])
        j = pred[j]
        while j != source:
            i = pred[j]
            path.append(edge_map[(i, j)])
            j = pred[j]
        edge_paths.append(path)

    return edge_paths
