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

from dyntapy.csr import UI32CSRMatrix
from dyntapy.supply_data import build_network



# store link_ids as tuple, infer from link_ids, to_node and from_node
@njit
def _make_out_links(links_to_include, from_node, to_node, tot_nodes):
    """

    creates out_links as dictionary

    Parameters
    ----------
    links_to_include: numpy.ndarray
        bool, 1D
    from_node: numpy.ndarray
        int, 1D
    to_node: numpy.ndarray
        int, 1D
    tot_nodes: int

    Returns
    -------
    out_links: numba.typed.Dict

    Notes
    -----

    It is more efficient to work with dyntapy's sparse matrices, however
    for some algorithms like Dial's Algorithm B editable graph structures are required.

    Examples
    --------
    the resulting dictionary gives access to the graph's structure as shown below

    >>> out_links[node_id]
    [node_id, link_id]

    """
    # creates out_links as dictionary, outlink[node] = [node_id, link_id]
    out_links = Dict()

    star_sizes = np.zeros(tot_nodes, dtype=np.int64)

    for link, include in enumerate(links_to_include):
        i = from_node[link]
        j = to_node[link]
        star_sizes[i] += 1

    max_out_links = np.max(star_sizes) # getting the max out_degree of nodes to limit
    # memory usage in creation
    star_sizes = np.zeros(tot_nodes, dtype=np.int64)
    for i in range(tot_nodes):
        out_links[i] = np.empty(
            (max_out_links, 2), dtype=np.int64
        )  # In traffic networks nodes should never have more than 10 outgoing edges..
    for link, include in enumerate(links_to_include):
        if include:
            i = from_node[link]
            j = to_node[link]
            out_links[i][star_sizes[i]][0] = j
            out_links[i][star_sizes[i]][1] = link
            star_sizes[i] += 1

    for i in range(tot_nodes):
        out_links[i] = out_links[i][: star_sizes[i]]
    return out_links


# store link_ids as tuple
@njit
def _make_in_links(links_to_include, from_node, to_node, tot_nodes):
    """

    creates in_links as dictionary.

    Parameters
    ----------
    links_to_include: numpy.ndarray
        bool, 1D
    from_node: numpy.ndarray
        int, 1D
    to_node: numpy.ndarray
        int, 1D
    tot_nodes: int

    Returns
    -------
    in_links: numba.typed.Dict

    Notes
    -----

    It is more efficient to work with dyntapy's sparse matrices, however
    for some algorithms like Dial's Algorithm B editable graph structures are required.

    Examples
    --------
    the resulting dictionary gives access to the graph's structure as shown below

    >>> in_links[node_id]
    [node_id, link_id]

    """
    backward_star = Dict()
    star_sizes = np.zeros(tot_nodes, dtype=np.int64)
    for link, include in enumerate(links_to_include):
        i = from_node[link]
        j = to_node[link]
        star_sizes[j] += 1
    max_in_links = np.max(star_sizes) # getting the max in_degree of nodes to limit
    # memory usage in creation
    for i in range(tot_nodes):
        backward_star[i] = np.empty(
            (max_in_links, 2), dtype=np.int64
        )

    star_sizes = np.zeros(tot_nodes, dtype=np.int64)  # , dtype=int_dtype
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


@njit()
def _get_link_id(from_node: int, to_node: int, out_links: UI32CSRMatrix):
    # convenience function to get link ids from sparse matrices
    for _id, j in zip(out_links.get_nnz(from_node), out_links.get_row(from_node)):
        if j == to_node:
            return _id
    # (from_node, to_node) not set in out_links
    raise AssertionError


@njit(nogil=True)
def pred_to_paths(predecessors, source, targets, out_links: UI32CSRMatrix):
    """
    converts optimal predecessor arrays to paths between source and targets

    Parameters
    ----------
    predecessors : numpy.ndarray
        int, 1D - predecessor of each node that is closest to `source`
    source: int
    targets: numpy.ndarray
    out_links: dyntapy.csr.UI32CSRMatrix

    Returns
    -------

    numba.typed.List

    """
    for target in targets:
        assert target != source
    link_paths = List()
    for j in targets:
        path = List()
        i = predecessors[j]
        path.append(_get_link_id(i, j, out_links))
        j = predecessors[j]

        while j != source:
            i = predecessors[j]
            path.append(_get_link_id(i, j, out_links))
            j = predecessors[j]
        link_paths.append(path)

    return link_paths

@njit(nogil=True)
def pred_to_path(predecessors, source, target, out_links: UI32CSRMatrix):
    """
    converts optimal predecessor arrays to path between source and target

    Parameters
    ----------
    predecessors : numpy.ndarray
        int, 1D - predecessor of each node that is closest to `source`
    source: int
    target: int
    out_links: dyntapy.csr.UI32CSRMatrix

    Returns
    -------

    numba.typed.List

    """
    j =target
    path = List()
    i = predecessors[j]
    path.append(_get_link_id(i, j, out_links))
    j = predecessors[j]

    while j != source:
        i = predecessors[j]
        path.append(_get_link_id(i, j, out_links))
        j = predecessors[j]

    return path

@njit(cache=True)
def dijkstra_all(
    costs,
    out_links: UI32CSRMatrix,
    source,
    is_centroid,
):
    """
    compiled one to all shortest path computation

    Parameters
    ----------
    costs: numpy.ndarray
        float, 1D
    out_links: dyntapy.csr.UI32CSRMatrix
    source: int
    is_centroid : numpy.ndarray
        bool, 1D

    Returns
    -------

    distances: numpy.ndarray
        float, 1D
    predecessors: numpy.ndarray
        int, 1D - predecessor for each node that is closest to source.
        Can be used to reconstruct paths.

    Examples
    --------

    A network object is build using `dyntapy.supply_data.build_network`, this happens
    during the intilization of any assignment.

    >>> network = dyntapy.supply_data.build_network(g)

    from this we can retrieve both `is_centroid` and `out_links`:

    >>> out_links = network.nodes.out_links
    >>> is_centroid = network.nodes.is_centroid

    which is what we need to run this function.


    See Also
    --------

    dyntapy.supply

    dyntapy.supply_data.build_network

    dyntapy.graph_utils.pred_to_paths

    """
    tot_nodes = out_links.tot_rows
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
    compiled one to many shortest path computation, terminates once distance array
    has been filled for all target nodes

    Parameters
    ----------
    costs: numpy.ndarray
        float, 1D
    out_links: dyntapy.csr.UI32CSRMatrix
    source: int
    is_centroid : numpy.ndarray
        bool, 1D
    targets: numpy.ndarray
        int, 1D

    Returns
    -------

    distances: numpy.ndarray
        float, 1D
    predecessors: numpy.ndarray
        int, 1D - predecessor for each node that is closest to source.

    Notes
    -----
    depending on how many targets there are to be found it can be faster to use
    `dyntapy.graph_utils.dijkstra_all`

    See Also
    --------

    dyntapy.graph_utils.pred_to_paths

    """
    tot_nodes = out_links.tot_rows
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


def get_all_shortest_paths(g, source, costs=None):
    """
    one to all shortest path computation

    Parameters
    ----------
    g: networkx.DiGraph
        as specified for assignments
    source: int
        node id
    costs: numpy.ndarray, optional
        if not set, free flow travel times are used based on defined length and speed


    Returns
    -------
    distances: numpy.ndarray
        float, 1D
    predecessors: numpy.ndarray
        int, 1D - predecessor for each node that is closest to source.

    See Also
    --------

    dyntapy.graph_utils.pred_to_paths

    Notes
    -----

    convenience function, the compiled functions cannot deal with branching on types

    """
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
    one to many shortest path computation

    Parameters
    ----------
    g: networkx.DiGraph
        as specified for assignments
    source: int
        node id
    targets: numpy.ndarray
        int, 1D
    costs: numpy.ndarray, optional
        if not set, free flow travel times are used based on defined length and speed
    return_paths: bool, optional
        set to true to get paths to each target

    Returns
    -------
    distances: numpy.ndarray
        float, 1D - distance to each target
    paths: list of list
        path to each target

    Notes
    -----

    convenience function, the compiled functions cannot deal with branching on types

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
