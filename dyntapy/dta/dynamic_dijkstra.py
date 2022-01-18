#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from heapq import heappop, heappush

import numpy as np
from numba import njit

from dyntapy.csr import UI32CSRMatrix


@njit(cache=True)
def dijkstra(costs, in_links: UI32CSRMatrix, target, tot_nodes, is_centroid):
    """
    typical dijkstra_with_targets implementation with heaps, fills the distances array with the results
    Parameters
    ----------
    is_centroid : bool array, dim tot_nodes, true if node is centroid, false otherwise
    tot_nodes : int, number of nodes
    costs : float32 vector
    out_links : CSR matrix, fromNode x Link
    target: integer ID of target node
    Returns
    -------
    distances: array 1D, dim tot_nodes. Distances from all nodes to the target node
    """
    # some minor adjustments from the static version to allow for the use of the csr structures
    # also removed conditional checks/ functionality that are not needed when this is integrated into route choice
    distances = np.full(tot_nodes, np.inf, dtype=np.float32)
    seen = np.copy(distances)
    my_heap = []
    seen[target] = np.float32(0)
    heap_item = (np.float32(0), np.float32(target))
    my_heap.append(heap_item)
    while my_heap:
        heap_item = heappop(my_heap)
        d = heap_item[0]
        i = np.uint32(heap_item[1])
        if distances[i] != np.inf:
            continue  # had this node already
        distances[i] = d
        if is_centroid[i] and not i == target:
            # centroids do not get unpacked, no connector routing..
            continue
        for in_link, j in zip(in_links.get_nnz(i), in_links.get_row(i)):
            ij_dist = distances[i] + costs[in_link]
            if seen[j] == np.inf or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (ij_dist, np.float32(j))
                heappush(my_heap, heap_item)
    return distances
