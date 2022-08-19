#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
import os
from itertools import count
from warnings import warn

import networkx as nx
import numpy as np
import osmnx as ox
import osmnx._errors
import pandas as pd
from numba.typed import List
from osmnx.distance import euclidean_dist_vec

from dyntapy.csr import UI32CSRMatrix, csr_prep, csr_sort
from dyntapy.settings import parameters
from dyntapy.supply import Links, Network, Nodes, Turns
from dyntapy.utilities import log

speed_mapping = parameters.supply.speed_mapping
cap_mapping = parameters.supply.cap_mapping
default_capacity = parameters.supply.cap_mapping
default_speed = parameters.supply.default_speed
default_buffer_dist = parameters.supply.default_buffer
turn_capacity_default = parameters.supply.turn_capacity_default
turn_type_default = parameters.supply.turn_type_default
node_capacity_default = parameters.supply.node_capacity_default
penalty_default = parameters.supply.turn_penalty_default
node_control_default = parameters.supply.node_control_default


def road_network_from_place(
    place,
    buffer_dist_close=default_buffer_dist,
    buffer_dist_extended=None,
):
    """
    retrieves road_network from OSM for driving.

    Detailed network for the inner polygon given by querying OSM.
    The buffer values determine surrounding polygons
    for which we acquire a coarser network.

    Parameters
    ----------
    place : str
        name of the city or region
    buffer_dist_close : float
        meters to buffer around, retain all roads with 'highway' = {'trunk', 'motorway',
        'primary'}
    buffer_dist_extended: float
        meters to buffer around, retain all roads with 'highway' = {'trunk', 'motorway',
        }


    Notes
    -----

    The filters for the buffers can be adjusted in the settings file.

    Returns
    -------

    networks.DiGraph
    """

    # TODO: provide default filters for network coarsity
    def acquire_graph():
        log("Starting to load from OSM")
        inner = ox.graph_from_place(place, network_type="drive")
        close = None
        extended = None
        try:
            close = ox.graph_from_place(
                place,
                network_type="drive",
                buffer_dist=buffer_dist_close,
                custom_filter=parameters.supply.close_surroundings_filter,
            )
        except (ox._errors.EmptyOverpassResponse, ValueError):
            warn("Could not find any elements in outer Buffer")
        try:
            extended = ox.graph_from_place(
                place,
                network_type="drive",
                buffer_dist=buffer_dist_extended,
                custom_filter=parameters.supply.extended_surroundings_filter,
            )
        except (ox._errors.EmptyOverpassResponse, ValueError):
            warn("Could not find any elements in inner Buffer")
        log("Finished downloading")
        print("composing")
        g = inner
        if close is not None:
            g = nx.compose(inner, close)
        if extended is not None:
            g = nx.compose(g, extended)

        return g

    g = acquire_graph()

    # osmnx generates MultiDiGraphs, meaning there can
    # be more than one edge connecting i ->j, a preliminary check on
    # them shows that these edges are mostly tagged with
    # (mildly) conflicting data, e.g. slightly different linestring
    # or length for simplification we just take all the
    # first entries, these differences will be mostly negligible
    # for us .. osm should straighten this one out ..
    dir_g = nx.DiGraph(
        g.edge_subgraph(
            [(u, v, k) for u, v, k in g.edges(keys=True) if k == 0 and u != v]
        )
    )  # do not allow circular edges..
    # in the remaining graph we have nodes that are only
    # weakly connected e.g. there is an edge (i,j) going away from
    # them ,but not towards them, or the other way around.
    # In that case if (i,j), or (i,j) respectively,
    # does not have a 'one_way':True we add the missing edge.
    # This way we can be sure that all nodes of the graph
    # build a strongly connected component.
    #
    components = list(nx.strongly_connected_components(dir_g))
    if len(components) > 1:
        largest = max(components, key=len)
        nodes_to_be_removed = []
        for c in components:
            if c == largest:
                pass
            else:
                for node in c:
                    nodes_to_be_removed.append(node)
        dir_g = nx.subgraph(dir_g, largest).copy()
    __clean_up_data(dir_g)
    dir_g = _convert_osm_to_gmns(dir_g)
    assert "crs" in dir_g.graph
    dir_g.graph["name"] = place
    log(
        f"retrieved network graph for {place},"
        f" with {dir_g.number_of_nodes()} nodes and {dir_g.number_of_edges()}"
        f" edges after processing",
        to_console=True,
    )

    return dir_g


def _convert_osm_to_gmns(g):
    # the attribute names are chosen in compliance with GMNS,
    # see https://github.com/zephyr-data-specs/
    # GMNS/blob/master/Specification/node.schema.json
    # and https://github.com/zephyr-data-specs/
    # GMNS/blob/master/Specification/link.schema.json
    # potentially could be extended to handle time of day changes in the future, see
    # https://github.com/zephyr-data-specs/
    # GMNS/blob/master/Specification/link_tod.schema.json
    # and also lanes:
    # https://github.com/zephyr-data-specs/
    # GMNS/blob/master/Specification/lane_tod.schema.json
    new_g = nx.DiGraph()
    edges = []
    for node, data in g.nodes.data():
        new_data = {
            "node_id": node,
            "x_coord": data["x"],
            "y_coord": data["y"],
            "node_type": None,
            "ctrl_type": None,
        }
        new_g.add_node(node, **new_data)
    for u, v, data in g.edges.data():
        new_data = {
            "from_node_id": u,
            "to_node_id": v,
            "length": data["length"],
            "free_speed": int(data["maxspeed"]),
            "facility_type": data["highway"],
            "lanes": int(data["lanes"]),
            "capacity": int(data["capacity"]),
        }
        if "geometry" in data:
            new_data["geometry"] = data["geometry"]
        if "name" in data:
            new_data["name"] = data["name"]
        else:
            if "ref" in data:
                new_data["name"] = data["ref"]
            else:
                new_data["name"] = None

        edges.append((u, v, new_data))
    new_g.add_edges_from(edges)
    new_g.graph = g.graph
    return new_g


def relabel_graph(g, return_inverse=False):
    """
    relabels graph's links and nodes consecutively starting from 0.

    Graphs obtained from external sources have labels that are often
    neither stable nor continuous. We relabel nodes and edges
    with our internal ids. The first C nodes in the network are centroids,
    with C the total number of centroids.
    The first K links in the network are source connectors,
    as link labelling is consecutive by the start node ids.
    Sink connector ids are therefore random.

    Parameters
    ----------
    g : networkx.DiGraph

    Returns
    -------
    new_g: networkx.Digraph
        with continuously labelled nodes, consistent with internal notation
    inverse: dict, optional
        a dictionary which maps each of the old node ids to the new ones


    """
    assert type(g) == nx.DiGraph
    centroids = [node for node, is_centroid in g.nodes.data("centroid") if is_centroid]
    intersection_nodes = [
        node for node, is_centroid in g.nodes.data("centroid") if not is_centroid
    ]
    centroids.sort()
    intersection_nodes.sort()
    new_g = nx.DiGraph()
    new_g.graph = g.graph
    link_counter = count(0)
    ordered_nodes = centroids + intersection_nodes
    for node_id, u in enumerate(ordered_nodes):
        _id = node_id
        data = g.nodes[u]
        new_g.add_node(_id, **data)
        new_g.nodes[_id][
            "ext_id"
        ] = u  # allows identifying the external labels later ..
        new_g.nodes[_id]["node_id"] = _id
        g.nodes[u]["node_id"] = _id
    for start_node, u in enumerate(ordered_nodes):
        _start_node = start_node
        for v in g.succ[u]:
            link_id = next(link_counter)
            end_node = g.nodes[v]["node_id"]
            data = g[u][v].copy()
            data["link_id"] = link_id
            data["from_node_id"] = _start_node
            data["to_node_id"] = end_node
            new_g.add_edge(_start_node, end_node, **data)
        # Note that the out_links of a given node always have consecutive ids
    log("graph relabeled")

    if return_inverse:
        inverse = {}
        for new, old in new_g.nodes.data("ext_id"):
            inverse[old] = new
        return new_g, inverse
    else:
        return new_g


def __clean_up_data(g: nx.DiGraph):
    for u, v, data in g.edges.data():
        tmp = set(data.keys()).intersection(
            {"capacity", "length", "maxspeed", "flow"}
        )  # filter keys for all traffic related float attr
        # maybe attach the list of used traffic keys to the graph?
        for key in tmp:
            try:
                data[key] = float(data[key])  # changing types of all relevant keys

            except (TypeError, ValueError) as e:
                # maxspeed can be all over the place in format
                # .. e.g. ['30', 'variable'] or ['none','100']
                assert key == "maxspeed"
                if isinstance(
                    data[key], list
                ):  # some tags, such as maxspeed may carry multiple values, if it's
                    # not a list but some other structure
                    # a new case should be defined to handle this  ..
                    float_list = []
                    for val in data[key]:
                        try:
                            float_list.append(float(val))
                        except ValueError:
                            pass
                    if not float_list:
                        # empty sequence
                        del data[key]
                    else:
                        data[key] = max(float_list)
                if isinstance(e, ValueError):
                    # maxspeed may be an individual string 'variable' or 'none' -
                    # we're just deleting this here to infer speed
                    # from the highway tag..
                    del data[key]

        if "maxspeed" not in data:  # if maxspeed is not tagged, we make a guess based
            # on the highway tag, see here
            # for more info see https://wiki.openstreetmap.org/wiki/Key:highway
            # this classification may vary from country to country ..
            # and was made without proper consideration
            try:
                data["maxspeed"] = __speed(data["highway"])
            except KeyError:
                # no highway tag..
                data["maxspeed"] = 50
        assert "length" in data
        data["length"] = data["length"] / 1000
        try:
            lanes = int(data["lanes"])
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, TypeError):
                assert isinstance(data["lanes"], list)
                try:
                    lanes = min([int(val) for val in data["lanes"]])
                except ValueError:
                    lanes = 1
            else:
                lanes = 1
        data["lanes"] = max(lanes, 1)
        data["capacity"] = __capacity(data["highway"], lanes)


def __capacity(highway_val, lanes):
    """
    capacity estimation based on Zilske, Michael, Andreas Neumann, and Kai Nagel.
    OpenStreetMap for traffic simulation. Technische Universität Berlin, 2015.
    This certainly needs refinement and updating but shall suffice for now..
    """
    if lanes == 0:
        lanes = 1
    try:
        if highway_val not in cap_mapping:
            cap_mapping[highway_val] = 1000

        return int(cap_mapping[highway_val] * lanes)
    except TypeError:
        # highway val is list..
        assert isinstance(highway_val, list)
        cap_list = [cap_mapping[item] for item in highway_val if item in cap_mapping]
        if len(cap_list) > 0:
            return min(cap_list) * lanes
        else:
            return default_capacity * lanes


def __speed(highway_val):
    try:
        if highway_val not in speed_mapping:
            speed_mapping[highway_val] = default_speed

        return speed_mapping[highway_val]
    except TypeError:
        # highway val is list..
        assert isinstance(highway_val, list)
        speed_list = [
            speed_mapping[item]
            for item in highway_val
            if speed_mapping[item] is not None
        ]
        if len(speed_list) > 0:
            return min(speed_list)
        else:
            return default_speed


def build_network(g, u_turns = False):
    """
    creates internal network representation

    Parameters
    ----------
    g: networkx.DiGraph
        road network graph

    Returns
    -------

    dyntapy.supply.Network


    See Also
    --------
    dyntapy.supply.Network

    """
    edge_data = [(_, _, data) for _, _, data in g.edges.data()]
    sorted_edges = sorted(edge_data, key=lambda t: t[2]["link_id"])
    sorted_nodes = sorted(g.nodes(data=True), key=lambda t: t[1]["node_id"])
    node_ids = np.array(
        [data["node_id"] for (_, data) in sorted_nodes], dtype=np.uint32
    )
    # for the future: remove this requirement of pre sorting of nodes.
    if not np.all(node_ids[1:] == node_ids[:-1] + 1):
        raise ValueError(
            "the node_ids in the graph are assumed to be"
            " monotonously increasing and have to be "
            "added accordingly"
        )
    tot_nodes = np.uint32(g.number_of_nodes())
    tot_links = np.uint32(g.number_of_edges())
    from_nodes = np.array(
        [d["from_node_id"] for (_, _, d) in sorted_edges], dtype=np.uint32
    )
    to_nodes = np.array([d["to_node_id"] for _, _, d in sorted_edges], dtype=np.uint32)
    link_ids = np.array([d["link_id"] for _, _, d in sorted_edges], dtype=np.uint32)
    if not np.all(link_ids[1:] == link_ids[:-1] + 1):
        raise ValueError(
            "the node_ids in the graph are assumed to be "
            "monotonously increasing and have to be "
            "added accordingly"
        )
    is_centroid = np.array(
        [bool(d.get("centroid")) for _, d in sorted_nodes], dtype=bool
    )
    x_coord = np.array([d["x_coord"] for (_, d) in sorted_nodes], dtype=np.float32)
    y_coord = np.array([d["y_coord"] for (_, d) in sorted_nodes], dtype=np.float32)
    nodes = _build_nodes(
        tot_nodes,
        tot_links,
        from_nodes,
        to_nodes,
        link_ids,
        is_centroid,
        x_coord,
        y_coord,
    )
    log("nodes passed")
    link_type = np.array(
        [np.int8(d.get("link_type", 0)) for (_, _, d) in sorted_edges], dtype=np.int8
    )
    turns = _build_turns(tot_nodes, nodes, link_type, u_turns = False)
    log("turns passed")

    link_capacity = np.array(
        [d["capacity"] for (_, _, d) in sorted_edges], dtype=np.float32
    )
    free_speed = np.array(
        [d["free_speed"] for (_, _, d) in sorted_edges], dtype=np.float32
    )
    lanes = np.array([d["lanes"] for (_, _, d) in sorted_edges], dtype=np.uint8)
    length = np.array([d["length"] for (_, _, d) in sorted_edges], dtype=np.float32)
    max_length = np.max(length)
    if np.max(length) > 100:
        warn(
            f"Network contains very long links, up to {max_length} km. "
            f"Implementation has not been verified for"
            f"this type of network. calculations may yield unexpected results."
        )

    # tot_connectors = (
    #     np.argwhere(link_type == 1).size + np.argwhere(link_type == -1).size
    # )
    # 1 is for sources (connectors leading out of a centroid)
    # -1 for sinks (connectors leading towards a centroid)
    links = _build_links(
        turns,
        tot_links,
        from_nodes,
        to_nodes,
        link_capacity,
        free_speed,
        lanes,
        length,
        link_type,
    )
    log("links passed")

    return Network(
        links,
        nodes,
        turns,
        g.number_of_edges(),
        g.number_of_nodes(),
        turns.capacity.size,
    )


def _build_nodes(
    tot_nodes, tot_links, from_nodes, to_nodes, link_ids, is_centroid, x_coord, y_coord
):
    values, col, row = csr_prep(
        np.column_stack((from_nodes, link_ids)), to_nodes, (tot_nodes, tot_links)
    )
    out_links = UI32CSRMatrix(values, col, row)
    values, col, row = csr_prep(
        np.column_stack((to_nodes, link_ids)), from_nodes, (tot_nodes, tot_links)
    )
    in_links = UI32CSRMatrix(values, col, row)
    capacity = np.full(tot_nodes, node_capacity_default, dtype=np.float32)
    control_type = np.full(tot_nodes, node_control_default, dtype=np.int8)
    # add boolean centroid array, alter control type(?) (if necessary)
    number_of_out_links = [
        len(in_links.get_row(row)) for row in np.arange(tot_nodes, dtype=np.uint32)
    ]
    number_of_in_links = [
        len(out_links.get_row(row)) for row in np.arange(tot_nodes, dtype=np.uint32)
    ]
    number_of_out_links = np.array(number_of_out_links, dtype=np.uint32)
    number_of_in_links = np.array(number_of_in_links, dtype=np.uint32)
    return Nodes(
        out_links,
        in_links,
        number_of_out_links,
        number_of_in_links,
        control_type,
        capacity,
        is_centroid,
        x_coord,
        y_coord,
    )


def _build_turns(tot_nodes, nodes: Nodes, link_types, u_turns =False):
    to_nodes = List()
    from_nodes = List()
    from_links = List()
    to_links = List()
    via_nodes = List()
    turn_counter = 0
    for via_node in np.arange(tot_nodes):
        # named here _attribute to indicate all the to nodes/links
        # that are associated with the via_node
        # turns are labelled here topologically by their respective
        # in_link labels, out_links are tiebreakers.

        _to_nodes = nodes.out_links.get_row(via_node)
        _from_nodes = nodes.in_links.get_row(via_node)
        _from_links = nodes.in_links.get_nnz(via_node)
        _to_links = nodes.out_links.get_nnz(via_node)
        for from_node, from_link in zip(_from_nodes, _from_links):
            for to_node, to_link in zip(_to_nodes, _to_links):
                if not (link_types[from_link] == -1 and link_types[to_link] == 1):
                    # always excluding turns that go from sink to source connectors
                    # and vice versa
                    if not u_turns:
                        if to_node==from_node:
                            continue
                    via_nodes.append(via_node)
                    to_nodes.append(to_node)
                    from_nodes.append(from_node)
                    from_links.append(from_link)
                    to_links.append(to_link)
                    turn_counter += 1
    fw_index_array = np.column_stack((from_links, to_links))
    turn_order = np.arange(turn_counter)
    res, turn_order = csr_sort(fw_index_array, turn_order, turn_counter)
    via_nodes = np.array(via_nodes, dtype=np.uint32)
    to_nodes = np.array(to_nodes, dtype=np.uint32)
    from_links = np.array(from_links, dtype=np.uint32)
    to_links = np.array(to_links, dtype=np.uint32)

    def sort(arr, order):
        tmp = np.empty_like(arr)
        for i, j in enumerate(order):
            tmp[i] = arr[j]
        return tmp

    via_nodes = sort(via_nodes, turn_order)
    from_nodes = sort(from_nodes, turn_order)
    to_nodes = sort(to_nodes, turn_order)
    from_links = sort(from_links, turn_order)
    to_links = sort(to_links, turn_order)
    number_of_turns = turn_counter
    penalty = np.full(number_of_turns, penalty_default, dtype=np.float32)
    capacity = np.full(number_of_turns, turn_capacity_default, dtype=np.float32)
    turn_type = np.full(number_of_turns, turn_type_default, dtype=np.int8)
    return Turns(
        penalty,
        capacity,
        np.array(from_nodes, dtype=np.uint32),
        np.array(via_nodes, dtype=np.uint32),
        np.array(to_nodes, dtype=np.uint32),
        np.array(from_links, dtype=np.uint32),
        np.array(to_links, dtype=np.uint32),
        turn_type,
    )


def _build_links(
    turns,
    tot_links,
    from_nodes,
    to_nodes,
    capacity,
    free_speed,
    lanes,
    length,
    link_type,
):
    """
    initiates all the different numpy arrays for the links object from networkx.DiGraph,
    requires the networkx graph to be set up as specified in the network_data
    Returns
    links : Links
    -------

    """
    # todo: iltm still working with this removed .. ?
    # length[length < 0.05] = 0.05
    tot_turns = np.uint32(len(turns.to_link))
    fw_index_array = np.column_stack(
        (turns.from_link, np.arange(tot_turns, dtype=np.uint32))
    )
    bw_index_array = np.column_stack(
        (turns.to_link, np.arange(tot_turns, dtype=np.uint32))
    )
    val = turns.to_link
    val, col, row = csr_prep(
        fw_index_array, val, (tot_links, tot_turns), unsorted=False
    )
    out_turns = UI32CSRMatrix(val, col, row)
    val = np.copy(turns.from_link)
    val, col, row = csr_prep(bw_index_array, val, (tot_links, tot_turns))
    in_turns = UI32CSRMatrix(val, col, row)

    return Links(
        length,
        from_nodes,
        to_nodes,
        capacity,
        free_speed,
        out_turns,
        in_turns,
        lanes,
        link_type,
    )


def get_toy_network(name):
    """
    retrieves toy network by name.

    Options are:
    'cascetta','simple_merge', 'simple_diverge', 'simple_bottleneck',
    'chicagosketch' 'chicagoregional' 'siouxfalls' 'birmingham'


    Parameters
    ----------
    name : str

    Returns
    -------
    networkx.DiGraph

    References
    ----------

    The source of 'chicagosketch' 'chicagoregional' 'siouxfalls' 'birmingham'
    is Ben Stabler et al. see : https://github.com/bstabler/TransportationNetworks

    The 'cascetta' network is from:

    Cascetta, Ennio. Transportation systems analysis: models and applications.
    Vol. 29. Springer Science & Business Media, 2009. Page 304.

    The remaining networks were set up by the authors.

    """

    g = nx.DiGraph()
    if name == "cascetta":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": np.sqrt(2)}),
            (2, {"x_coord": np.sqrt(2) + 2, "y_coord": 2 * np.sqrt(2)}),
            (3, {"x_coord": np.sqrt(2) + 2, "y_coord": 0}),
            (4, {"x_coord": 2 * np.sqrt(2) + 2, "y_coord": np.sqrt(2)}),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        ebunch_of_edges = [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 3),
            (4, 2),
            (3, 2),
            (3, 1),
            (2, 1),
        ]
        bottle_neck_edges = [(2, 3), (3, 2), (3, 4), (4, 3)]
        g.add_edges_from(ebunch_of_edges)
        _set_toy_network_attributes(g, bottle_neck_edges)

    elif name == "simple_bottleneck":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 1}),
            (2, {"x_coord": 3, "y_coord": 1}),
            (3, {"x_coord": 4, "y_coord": 1}),
            (4, {"x_coord": 5, "y_coord": 1}),
        ]
        ebunch_of_edges = [(2, 3), (3, 2), (1, 2), (2, 1), (3, 4), (4, 3)]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(2, 3), (3, 2)]
        _set_toy_network_attributes(g, bottleneck_edges)
    elif name == "simple_merge":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 1}),
            (2, {"x_coord": 2, "y_coord": 2}),
            (3, {"x_coord": 4, "y_coord": 1.5}),
            (4, {"x_coord": 5, "y_coord": 1.5}),
            (5, {"x_coord": 6, "y_coord": 1.5}),
        ]
        ebunch_of_edges = [
            (2, 3),
            (3, 2),
            (1, 3),
            (3, 1),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(3, 4), (4, 3)]
        _set_toy_network_attributes(g, bottleneck_edges)
    elif name == "simple_diverge":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 2}),
            (2, {"x_coord": 3, "y_coord": 2}),
            (3, {"x_coord": 4, "y_coord": 2}),
            (4, {"x_coord": 5, "y_coord": 1}),
            (5, {"x_coord": 5, "y_coord": 3}),
        ]
        ebunch_of_edges = [
            (2, 3),
            (3, 2),
            (1, 2),
            (2, 1),
            (3, 4),
            (4, 3),
            (3, 5),
            (5, 3),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(2, 3), (3, 2)]
        _set_toy_network_attributes(g, bottleneck_edges)
    elif name in [
        "chicagosketch",
        "chicagoregional",
        "siouxfalls",
        "birmingham",
    ]:
        # The source of these networks is Ben Stabler et al.,
        # see here: https://github.com/bstabler/TransportationNetworks
        # don't forget to cite as shown in the repository

        edge_file = "{0}{1}{2}_net.tntp".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.sep + "toy_networks" + os.path.sep,
            name,
        )
        node_file = "{0}{1}{2}_node.tntp".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.sep + "toy_networks" + os.path.sep,
            name,
        )
        edge_df = pd.read_csv(edge_file, skiprows=8, sep="\t")
        if name == "birmingham":
            sep = "       "
        else:
            sep = "\t"
        node_df = pd.read_csv(node_file, sep=sep)
        Graphtype = nx.DiGraph()
        edge_df["init_node"] = edge_df["init_node"] - 1
        edge_df["term_node"] = edge_df["term_node"] - 1
        edge_df["free_speed"] = edge_df["length"] / edge_df["free_flow_time"]
        edge_df["lanes"] = 1
        g = nx.from_pandas_edgelist(
            edge_df,
            source="init_node",
            target="term_node",
            edge_attr=["length", "capacity", "free_speed", "lanes"],
            create_using=Graphtype,
        )
        node_df = node_df.rename(columns={col: col.lower() for col in node_df.columns})
        node_df["node"] = node_df["node"] - 1
        for node, x, y in zip(node_df["node"], node_df["x"], node_df["y"]):
            try:
                g.nodes[node]["x_coord"] = x
                g.nodes[node]["y_coord"] = y
            except KeyError:
                #   no edge with this node was added..
                continue
        g.graph = {"name": name}
        return g

    else:
        raise ValueError("no toy network provided under that name")
    g.graph = {"name": name}
    return g


def _set_toy_network_attributes(g, bottleneck_edges):
    capacity = 2000
    free_speed = 80
    lanes = 1
    bottleneck_capacity = 500
    bottleneck_free_speed = 120
    node_ctrl_type = "none"
    for v in g.nodes:
        g.nodes[v]["ctrl_type"] = node_ctrl_type
    for u, v, data in g.edges.data():
        y1 = g.nodes[v]["y_coord"]
        x1 = g.nodes[v]["x_coord"]
        y0 = g.nodes[u]["y_coord"]
        x0 = g.nodes[u]["x_coord"]
        data["length"] = euclidean_dist_vec(y0, x0, y1, x1)
        data["capacity"] = capacity
        data["free_speed"] = free_speed
        data["lanes"] = lanes
        if (u, v) in bottleneck_edges:
            data["capacity"] = bottleneck_capacity
            data["free_speed"] = bottleneck_free_speed
