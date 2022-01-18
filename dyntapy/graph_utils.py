#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from heapq import heappop, heappush

import networkx as nx
import numpy as np
from numba import generated_jit, njit, typeof, types
from numba.core.errors import TypingError
from numba.typed import Dict, List

from dyntapy.csr import UI32CSRMatrix, f32csr_type, ui8csr_type, ui32csr_type
from dyntapy.supply_data import build_network


# store link_ids as tuple, infer from link_ids, to_node and from_node
@njit
def make_out_links(links_to_include, from_node, to_node, tot_nodes):
    # creates out_links as dictionary, outlink[node] = [node_id, link_id]
    forward_star = Dict()
    star_sizes = np.zeros(tot_nodes, dtype=np.int64)
    for i in range(tot_nodes):
        forward_star[i] = np.empty(
            (20, 2), dtype=np.int64
        )  # In traffic networks nodes will never have more than 10 outgoing edges..
    for link, include in enumerate(links_to_include):
        if include:
            i = from_node[link]
            j = to_node[link]
            forward_star[i][star_sizes[i]][0] = j
            forward_star[i][star_sizes[i]][1] = link
            star_sizes[i] += 1

    for i in range(tot_nodes):
        forward_star[i] = forward_star[i][: star_sizes[i]]
    return forward_star


# store link_ids as tuple
@njit
def make_in_links(links_to_include, from_node, to_node, tot_nodes):
    # creates in_links as dictionary, in_links[node] = [node_id, link_id]
    # equivalent to make_out_links but with swapped from and to_nodes values
    # left hear for clarity
    backward_star = Dict()
    star_sizes = np.zeros(tot_nodes, dtype=np.int64)  # , dtype=int_dtype
    for i in range(tot_nodes):
        backward_star[i] = np.empty(
            (20, 2), dtype=np.int64
        )  # nodes in traffic networks have less than 10 outgoing edges..
    for link, include in enumerate(links_to_include):
        if include:
            i = from_node[link]
            j = to_node[link]
            # print(f'i: {i} j: {j} ')
            backward_star[j][star_sizes[j]][0] = i
            backward_star[j][star_sizes[j]][1] = link
            star_sizes[j] += 1
    for i in range(tot_nodes):
        backward_star[i] = backward_star[i][: star_sizes[i]]
    return backward_star


def node_to_edge_path(node_path, edge_map):
    edge_path = List()
    for id, node in enumerate(node_path[:-1]):
        edge_path.append(edge_map[(node, node_path[id + 1])])
    return edge_path


@njit()
def get_link_id(from_node: int, to_node: int, out_links: UI32CSRMatrix):
    # convenience function to get link ids from sparse matrices
    for _id, j in zip(out_links.get_nnz(from_node), out_links.get_row(from_node)):
        if j == to_node:
            return _id
    # (from_node, to_node) not set in out_links
    raise AssertionError


@njit(nogil=True)
def pred_to_paths(predecessors, source, targets, out_links: UI32CSRMatrix):
    for target in targets:
        assert target != source
    link_paths = List()
    for j in targets:
        path = List()
        i = predecessors[j]
        path.append(get_link_id(i, j, out_links))
        j = predecessors[j]

        while j != source:
            i = predecessors[j]
            path.append(get_link_id(i, j, out_links))
            j = predecessors[j]
        link_paths.append(path)

    return link_paths


@njit(cache=True)
def dijkstra_all(
    costs,
    out_links: UI32CSRMatrix,
    source,
    is_centroid,
):
    """
    typical dijkstra_with_targets implementation with heaps, fills the distances
    array with the results
    Parameters
    ----------
    is_centroid : bool array, dim tot_nodes, true if node is centroid, false otherwise
    tot_nodes : int, number of nodes
    costs : float32 vector
    out_links : CSR matrix, fromNode x Link
    Returns
    -------
    distances: array 1D, dim tot_nodes. Distances from all nodes to the target node
    """
    # some minor adjustments from the static version to allow for the use of the csr
    # structures
    # also removed conditional checks/ functionality that are not needed when this is
    # integrated into route choice
    tot_nodes = out_links.get_nnz_rows().size
    distances = np.full(tot_nodes, np.inf, dtype=np.float32)
    predecessors = np.empty(tot_nodes, dtype=np.uint32)
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
        if is_centroid[i] and not i == source:
            # centroids do not get unpacked, no connector routing..
            continue
        for out_link, j in zip(out_links.get_nnz(i), out_links.get_row(i)):
            ij_dist = distances[i] + costs[out_link]
            if seen[j] == np.inf or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (np.float32(ij_dist), np.float32(j))
                heappush(my_heap, heap_item)
                predecessors[j] = i
        distances[source] = 0
    return distances, predecessors


@njit(cache=True)
def dijkstra_with_targets(
    costs, out_links: UI32CSRMatrix, source, is_centroid, targets
):
    """
    typical dijkstra_with_targets implementation with heaps, fills the distances
    array with the results
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
    # some minor adjustments from the static version to allow for the use of the csr
    # structures
    # also removed conditional checks/ functionality that are not needed when this is
    # integrated into route choice
    tot_nodes = out_links.get_nnz_rows().size
    distances = np.full(tot_nodes, np.inf, dtype=np.float32)
    predecessors = np.empty(tot_nodes, dtype=np.uint32)
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
        if is_centroid[i] and not i == source:
            # centroids do not get unpacked, no connector routing..
            continue
        for index, target in enumerate(targets):
            if i == target:
                targets = np.delete(targets, index)
                break
        if targets.size == 0:
            break
        for out_link, j in zip(out_links.get_nnz(i), out_links.get_row(i)):
            ij_dist = distances[i] + costs[out_link]
            if seen[j] == np.inf or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (np.float32(ij_dist), np.float32(j))
                heappush(my_heap, heap_item)
                predecessors[j] = i
    return distances, predecessors


def get_all_shortest_paths(
    g: nx.DiGraph, source: int, costs: np.ndarray = None
) -> (np.ndarray, np.ndarray):
    if costs is not None:
        if not isinstance(costs, np.ndarray):
            raise TypeError
        if not costs.size == g.number_of_edges():
            raise ValueError
    if not isinstance(source, int):
        raise TypeError
    # transform networkx DiGraph to internal network presentation
    network = build_network(g)
    if costs is None:
        # assuming free flow travel times
        costs = network.links.length / network.links.free_speed
        costs = costs.astype(np.float32)
    out_links = network.nodes.out_links
    distances, pred = dijkstra_all(costs, out_links, source, network.nodes.is_centroid)
    return distances, pred


def get_shortest_paths(g, source, targets, costs=None, return_paths=False):
    """

    Parameters
    ----------
    g :
    source :
    targets :
    costs :

    Returns
    -------

    """
    if not isinstance(targets, (np.ndarray, list)):
        raise TypeError
    else:
        if isinstance(targets, list):
            if not all(isinstance(k, int) for k in targets):
                raise TypeError
        targets = np.array(targets, dtype=np.uint32)
    if source in targets:
        raise ValueError("targets cannot contain source")
    if costs is not None:
        if not isinstance(costs, np.ndarray):
            raise TypeError
        if not costs.size == g.number_of_edges():
            raise ValueError
    if not isinstance(source, int):
        raise TypeError
    # transform networkx DiGraph to internal network presentation
    network = build_network(g)
    if costs is None:
        # assuming free flow travel times
        costs = network.links.length / network.links.free_speed
        costs = costs.astype(np.float32)
    out_links = network.nodes.out_links
    distances, pred = dijkstra_with_targets(
        costs, out_links, source, network.nodes.is_centroid, targets=targets
    )
    if return_paths:
        path_costs = np.array([distances[target] for target in targets])
        paths = pred_to_paths(pred, source, targets, out_links)
        # returned paths are numba typed Lists, copying needed. Otherwise they cannot
        # be processed like you would do with regular python lists.
        paths = [[link for link in path] for path in paths]
        return path_costs, paths
    else:
        path_costs = np.array([distances[target] for target in targets])
        return path_costs
