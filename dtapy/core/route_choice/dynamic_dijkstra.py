#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from heapq import heappush, heappop
import numpy as np
from dtapy.datastructures.csr import UI32CSRMatrix


@njit
def dijkstra(costs, out_links: UI32CSRMatrix, source, tot_nodes):
    """
    typical dijkstra implementation with heaps, fills the distances array with the results
    Parameters
    ----------
    tot_nodes : int, number of nodes
    costs : float32 vector
    out_links : CSR matrix, fromNode x Link
    source: integer ID of source node

    Returns
    -------
    distances: array 1D, dim tot_nodes
    """
    # some minor adjustments from the static version to allow for the use of the csr structures
    # also removed conditional checks/ functionality that are not needed when this is integrated into route choice
    distances = np.full(tot_nodes, np.inf, dtype=np.float32)
    seen = np.copy(distances)
    my_heap = []
    seen[source] = np.float32(0)
    heap_item = (np.float32(0), np.float32(source))
    my_heap.append(heap_item)
    while my_heap:
        heap_item = heappop(my_heap)
        d = heap_item[0]
        i = np.uint32(heap_item[1])
        if distances[i] != np.inf:
            continue  # had this node already
        distances[i] = d
        for out_link, j in zip(out_links.get_nnz(i), out_links.get_row(i)):
            ij_dist = distances[i] + costs[out_link]
            if seen[j] == np.inf or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (ij_dist, np.float32(j))
                heappush(my_heap, heap_item)
    return distances
