#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from heapq import heappop, heappush
import numpy as np
from numba import float64, njit
from numba.typed import Dict, List

from dyntapy.csr import UI32CSRMatrix
from dyntapy.supply_data import build_network


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

    """
    out_links = Dict()

    star_sizes = np.zeros(tot_nodes, dtype=np.int64)

    for link, include in enumerate(links_to_include):
        i = from_node[link]
        j = to_node[link]
        star_sizes[i] += 1

    max_out_links = np.max(star_sizes)  # getting the max out_degree of nodes to limit
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

    """
    backward_star = Dict()
    star_sizes = np.zeros(tot_nodes, dtype=np.int64)
    for link, include in enumerate(links_to_include):
        i = from_node[link]
        j = to_node[link]
        star_sizes[j] += 1
    max_in_links = np.max(star_sizes)  # getting the max in_degree of nodes to limit
    # memory usage in creation
    for i in range(tot_nodes):
        backward_star[i] = np.empty((max_in_links, 2), dtype=np.int64)

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
def pred_to_paths(
    predecessors, source, targets, out_links: UI32CSRMatrix, reverse=False
):
    """
    converts optimal predecessor arrays to paths between source and targets

    Parameters
    ----------
    predecessors : numpy.ndarray
        int, 1D - predecessor of each node that is closest to `source`
    source: int
    targets: numpy.ndarray
    out_links: dyntapy.csr.UI32CSRMatrix
    reverse: bool, default True
        if predecessors array is a successor array.


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
        if not reverse:

            path.insert(0, _get_link_id(i, j, out_links))
        else:
            path.insert(0, _get_link_id(j, i, out_links))
        j = predecessors[j]

        while j != source:
            i = predecessors[j]
            if not reverse:
                path.insert(0, _get_link_id(i, j, out_links))
            else:
                path.insert(0, _get_link_id(j, i, out_links))
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

    j = target

    path = List()

    i = predecessors[j]

    path.append(_get_link_id(i, j, out_links))

    j = predecessors[j]

    while j != source:
        i = predecessors[j]

        path.insert(0, _get_link_id(i, j, out_links))

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

    costs : numpy.ndarray
        float, 1D
    out_links : dyntapy.csr.UI32CSRMatrix
    source: int
    is_centroid : numpy.ndarray
        bool, 1D

    Returns
    -------

    distances: numpy.ndarray
        float, 1D
    predecessors: numpy.ndarray
        int, 1D - predecessor for each node that is closest to source.

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
    has been filled for all target nodes.

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


@njit
def _kspwlo_esx(
    costs,
    out_links,
    source,
    target,
    k,
    is_centroid,
    sim_threshold,
    detour_rejection=0.5,
):
    # Chondrogiannis, Theodoros, et al. "Finding k-shortest paths with limited
    # overlap." The VLDB Journal 29.5 (2020): 1023-1047.

    # esx minW variant implementation, determines k-shortest paths of minimum length
    # with an overlap smaller than sim_threshold heuristically
    _targets = np.empty(1)
    _targets[0] = target  # making singular target iterable to comply with fixed typed
    # of dijkstra_with_targets arg
    distances, pred = dijkstra_with_targets(
        costs, out_links, source, is_centroid, _targets
    )
    sp = pred_to_path(pred, source, target, out_links)
    do_not_ignore = np.full(costs.size, False)
    solution_paths = List()
    solution_paths.append(sp)
    path_rejection_cost = (1 + detour_rejection) * distances[target]
    path_lengths = List()
    path_lengths.append(distances[target])
    links_to_ignore = np.full(costs.size, False)
    min_queue = List()
    min_queue.append((0.0, 0.0))  # numba typing ..
    min_queue.pop()
    for link in sp:
        heappush(min_queue, (float64(costs[link]), float64(link)))  # minW variant

    path_queues = List()
    path_queues.append(min_queue)  # each path that's part of the solution has its own
    # queue of edges to be considered for removal
    incomplete_solution = False
    while len(solution_paths) < k and len(min_queue) > 0:
        candidate_path = solution_paths[-1]
        if incomplete_solution:
            break
        while True:
            # finding the max sim path with a non-empty queue
            # print(f'There are {len(solution_paths)} solutions')
            # print(f'there are {len(path_queues)} queues')
            # print(f'{solution_paths[-1]=}')
            # print(f'calling max_similarity, {candidate_path=}')
            max_sim, max_path, path_idx = max_similarity(
                candidate_path, solution_paths, path_queues
            )

            # print(f'path has {max_sim=}')
            path_queue = path_queues[path_idx]
            if path_idx == -1:
                incomplete_solution = True
                print("incomplete solution")
                break  # no edges can be removed, less than k paths found
            if max_sim < sim_threshold:
                # candidate can be added to the set
                #   print('candidate can be added to the set!')
                break

            heap_item = heappop(path_queue)
            link = np.uint32(heap_item[1])
            # print(f'{link=} being removed')
            # print('check link in do ignore')
            if do_not_ignore[link]:
                continue
            links_to_ignore[link] = True
            # new candidate path with links to ignore
            # print(f'calling sp s')
            path_found, dist, tentative_path = _dijkstra_with_target_ignored_links(
                costs,
                out_links,
                source,
                is_centroid,
                target,
                links_to_ignore,
                path_rejection_cost,
            )
            if not path_found:
                #   print(f'{link=} cannot be ignored')
                do_not_ignore[link] = True
                links_to_ignore[link] = False  # needed for connectivity
            else:
                # proposed path is valid
                candidate_path = tentative_path
            #   print(f'candidate path replaced with tentative path')

        max_sim, max_path, path_idx = max_similarity(
            candidate_path, solution_paths, path_queues
        )
        if max_sim < sim_threshold:
            # print('new path added to solution set')
            solution_paths.append(candidate_path)
            path_lengths.append(dist)
            # initializing queue for next iteration

            new_path_queue = List()
            new_path_queue.append((0.0, 0.0))
            new_path_queue.pop()
            for link in candidate_path:
                heappush(new_path_queue, (float64(costs[link]), float64(link)))
            path_queues.append(new_path_queue)
    return solution_paths, path_lengths


@njit
def max_similarity(path, path_set, path_queues):
    # similarity is defined as the number of shared edges divided by the min length
    # in number of edges of
    # the compared paths. Restricted to [0,1].
    # returns the max_similarity path for which there are items in the queue
    max_sim = 0
    max_sim_path = path
    max_idx = -1
    for idx, alternative in enumerate(path_set):
        min_path_length = min(len(path), len(alternative))
        shared_edges = 0
        for link in path:
            if link in alternative:
                shared_edges += 1
        sim = shared_edges / min_path_length
        if sim > max_sim:
            max_sim = sim
            if len(path_queues[idx]) > 0:
                max_sim_path = alternative
                max_idx = idx
    return max_sim, max_sim_path, max_idx


@njit
def _dijkstra_with_target_ignored_links(
    costs,
    out_links: UI32CSRMatrix,
    source,
    is_centroid,
    target,
    links_to_ignore,
    max_dist,
):
    # TODO: add links to ignore option

    tot_nodes = out_links.tot_rows
    distances = np.full(tot_nodes, np.inf, dtype=np.float32)
    predecessors = np.empty(tot_nodes, dtype=np.uint32)
    seen = np.copy(distances)
    my_heap = []
    seen[source] = np.float32(0)
    heap_item = (np.float32(0), np.float32(source))
    my_heap.append(heap_item)
    target_found = False
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
        if i == target:
            target_found = True
            break
        for out_link, j in zip(out_links.get_nnz(i), out_links.get_row(i)):
            if links_to_ignore[out_link]:
                continue
            ij_dist = distances[i] + costs[out_link]
            if seen[j] == np.inf or ij_dist < seen[j]:
                seen[j] = ij_dist
                heap_item = (np.float32(ij_dist), np.float32(j))
                heappush(my_heap, heap_item)
                predecessors[j] = i
    if target_found and distances[target] < max_dist:
        path = pred_to_path(predecessors, source, target, out_links)
    else:
        target_found = False
        path = List()
        path.append(np.uint32(10000000))
    return target_found, distances[target], path


def kspwlo_esx(
    costs,
    out_links,
    source,
    target,
    k,
    is_centroid,
    sim_threshold,
    detour_rejection=0.5,
):
    """

    computes k-shortest paths with a maximum overlap of `sim_threshold`.

    Parameters
    ----------

    costs: numpy.ndarray
        float, 1D - cost for each link
    out_links: dyntapy.csr.UI32CSRMatrix
        adjacency structure
    source: int
        source node
    target: int
        target node
    k: int
        number of paths to generate
    is_centroid: np.ndarray
        bool, 1D - centroids are ignored for routing
    sim_threshold: float
        threshold for similarity between paths in the solution set, [0,1]
    detour_rejection: float
        path quality criteria

    Returns
    -------

    solution_paths: list of list
        each entry is a solution path
    path_lengths : list
        length of each solution path as the sum of traversed link costs

    Notes
    -----

    `detour_rejection` has been added by the developers to prune bad solutions.
    A value of 0.10 indicates that paths can be at most 10 percent worse than the
    shortest path solution. Similar to a lower `sim_threshold` this setting may
    affect the completeness of the results, see [5]_.

    References
    ----------

    .. [5] Chondrogiannis, Theodoros, Panagiotis Bouros, Johann Gamper, Ulf Leser, and David B. Blumenthal. ‘Finding K-Shortest Paths with Limited Overlap’. The VLDB Journal 29, no. 5 (1 September 2020): 1023–47.  https://doi.org/10.1007/s00778-020-00604-x.

    """
    solution_paths, distances = _kspwlo_esx(
        costs,
        out_links,
        source,
        target,
        k,
        is_centroid,
        sim_threshold,
        detour_rejection,
    )
    py_solution_paths = []
    py_distances = []
    for path, dist in zip(solution_paths, distances):
        py_path = []
        for link in path:
            py_path.append(link)
        py_solution_paths.append(py_path)
        py_distances.append(dist)
    return py_solution_paths, py_distances


def get_k_shortest_paths(
    g, source, target, costs=None, k=3, sim_threshold=0.75, detour_rejection=0.5
):
    """
    computes k-shortest paths with a maximum overlap of `sim_threshold`

    Parameters
    ----------
    g: networkx.DiGraph
        as specified for assignments
    source: int
        node id
    target: int
        node id
    costs: numpy.ndarray, optional
        if not set, free flow travel times are used based on defined length and speed
    k: int
        number of the shortest paths to return
    sim_threshold: float
        threshold for similarity between paths in the solution set, [0,1]
    detour_rejection: float
        path quality criteria

    Returns
    -------
    solution_paths: list of list
        each entry is a solution path
    path_lengths : list
        length of each solution path as the sum of traversed link costs

    Notes
    -----

    `detour_rejection` has been added by the developers to prune bad solutions.
    A value of 0.10 indicates that paths can be at most 10 percent worse than the
    shortest path solution. Similar to a lower `sim_threshold` this setting may
    affect the completeness of the results.

    See Also
    ----------

    dyntapy.graph_utils.kspwlo_esx

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
    solution_paths, distances = kspwlo_esx(
        costs,
        out_links,
        source,
        target,
        k,
        network.nodes.is_centroid,
        sim_threshold,
        detour_rejection,
    )

    return solution_paths, distances
