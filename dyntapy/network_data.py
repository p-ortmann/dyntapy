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
import networkx as nx
import osmnx as ox
from dyntapy.settings import dynamic_parameters
from dyntapy.utilities import log, _filepath
from itertools import count
import numpy as np

speed_mapping = dynamic_parameters.supply.speed_mapping
cap_mapping = dynamic_parameters.supply.cap_mapping
default_capacity = dynamic_parameters.supply.cap_mapping
default_speed = dynamic_parameters.supply.default_speed


def get_from_ox_and_save(name: str, reload=False):
    file_path = _filepath(name + '_pure_ox')
    if not reload:
        try:
            g = nx.read_gpickle(file_path)
        except FileNotFoundError:
            log(f'city {name} could not be found in data folder, loading from osm', level=50)
            g = ox.graph_from_place(name, network_type='drive')
            nx.write_gpickle(g, file_path)
    else:
        log('Starting to load from OSM')
        g = ox.graph_from_place(name, network_type='drive')
        log('Finished downloading')
        nx.write_gpickle(g, file_path)
    # osmnx generates MultiDiGraphs, meaning there can be more than one edge connecting i ->j, a preliminary check on
    # them shows that these edges are mostly tagged with (mildly) conflicting data, e.g. slightly different linestring
    # or length for simplification we just take all the first entries, these differences will be mostly negligible
    # for us .. osm should straighten this one out ..
    dir_g = nx.DiGraph(g.edge_subgraph(
        [(u, v, k) for u, v, k in g.edges(keys=True) if k == 0 and u != v]))  # do not allow circular edges..
    # in the remaining graph we have nodes that are only weakly connected e.g. there is an edge (i,j) going away from
    # them ,but not towards them, or the other way around. In that case if (i,j), or (i,j) respectively,
    # does not have a 'one_way':True we add the missing edge. This way we can be sure that all nodes of the graph
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
        deleted = nx.subgraph(dir_g, nodes_to_be_removed)
        dir_g = nx.subgraph(dir_g, largest).copy()
    else:
        deleted = None
    __clean_up_data(dir_g)
    dir_g = convert_ox_to_gmns(dir_g)
    nx.write_gpickle(dir_g, _filepath(name + "_gmns"))
    assert 'crs' in dir_g.graph
    dir_g.graph['name'] = name
    log(f'retrieved network graph for {name},'
        f' with {dir_g.number_of_nodes()} nodes and {dir_g.number_of_edges()} edges after processing', to_console=True)
    return dir_g


def convert_ox_to_gmns(g):
    # the attribute names are chosen in compliance with GMNS,
    # see https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/node.schema.json
    # and https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/link.schema.json
    # potentially could be extended to handle time of day changes in the future, see
    # https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/link_tod.schema.json
    # and also lanes:
    # https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/lane_tod.schema.json
    new_g = nx.MultiDiGraph()
    edges = []
    for node, data in g.nodes.data():
        new_data = {'node_id': node, 'x_coord': data['x'], 'y_coord': data['y'], 'node_type': None, 'ctrl_type': None}
        new_g.add_node(node, **new_data)
    for u, v, data in g.edges.data():
        new_data = {'from_node_id': u, 'to_node_id': v, 'length': data['length'], 'free_speed': int(data['maxspeed']),
                    'facility_type': data['highway'], 'lanes': int(data['lanes']), 'capacity': int(data['capacity'])}
        if 'geometry' in data:
            new_data['geometry'] = data['geometry']
        if 'name' in data:
            new_data['name'] = data['name']
        else:
            if 'ref' in data:
                new_data['name'] = data['ref']
            else:
                new_data['name'] = None

        edges.append((u, v, new_data))
    new_g.add_edges_from(edges)
    new_g.graph = g.graph
    return new_g


def relabel_graph(g):
    """
    osmnx labels the graph nodes and edges by their osm ids. These are neither stable nor continuous. We relabel nodes and edges
    with our internal ids. The first C nodes in the network are centroids, with C the total number of centroids.
    The first K links in the network are source connectors, as link labelling is consecutive by the start node ids.
    Sink connector ids are therefore random.
    Parameters
    ----------
    g : nx.DiGraph
    Returns
    -------
    nx.Digraph with continuously labelled nodes, consistent with internal notation


    """
    centroids = [node for node, is_centroid in g.nodes.data('centroid') if is_centroid]
    intersection_nodes = [node for node, is_centroid in g.nodes.data('centroid') if not is_centroid]
    new_g = nx.MultiDiGraph()
    new_g.graph = g.graph
    link_counter = count(0)
    ordered_nodes = centroids + intersection_nodes
    for node_id, u in enumerate(ordered_nodes):
        _id = node_id
        data = g.nodes[u]
        new_g.add_node(_id, **data)
        new_g.nodes[_id]['ext_id'] = u  # allows identifying the external labels later ..
        new_g.nodes[_id]['node_id'] = _id
        g.nodes[u]['node_id'] = _id
    for start_node, u in enumerate(ordered_nodes):
        _start_node = start_node
        for v in g.succ[u]:
            for k in g[u][v].keys():
                link_id = next(link_counter)
                end_node = g.nodes[v]['node_id']
                data = g[u][v][k].copy()
                data['link_id'] = link_id
                data['from_node_id'] = _start_node
                data['to_node_id'] = end_node
                new_g.add_edge(_start_node, end_node, key=k, **data)
        # Note that the out_links of a given node always have consecutive ids
    log('graph relabeled')
    return new_g


def __clean_up_data(g: nx.DiGraph):
    for u, v, data in g.edges.data():
        tmp = set(data.keys()).intersection(
            {'capacity', 'length', 'maxspeed', 'flow'})  # filter keys for all traffic related float attr
        # maybe attach the list of used traffic keys to the graph?
        for key in tmp:
            try:
                data[key] = float(data[key])  # changing types of all relevant keys

            except (TypeError, ValueError) as e:
                # maxspeed can be all over the place in format .. e.g. ['30', 'variable'] or ['none','100']
                assert key == 'maxspeed'
                if isinstance(data[key], list):  # some tags, such as maxspeed may carry multiple values, if it's
                    # not a list but some other structure a new case should be defined to handle this  ..
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
                    # maxspeed may be an individual string 'variable' or 'none' - we're just deleting this here to infer speed from the highway tag..
                    del data[key]

        if 'maxspeed' not in data:  # if maxspeed is not tagged, we make a guess based on the highway tag, see here
            # for more info see https://wiki.openstreetmap.org/wiki/Key:highway
            # this classification may vary from country to country .. and was made without proper consideration
            try:
                data['maxspeed'] = __speed(data['highway'])
            except KeyError:
                # no highway tag..
                data['maxspeed'] = 50
        assert 'length' in data
        data['length'] = data['length'] / 1000
        try:
            lanes = int(data['lanes'])
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, TypeError):
                assert isinstance(data['lanes'], list)
                try:
                    lanes = min([int(val) for val in data['lanes']])
                except ValueError:
                    lanes = 1
            else:
                lanes = 1
        data['lanes'] = max(lanes,1)
        data['capacity'] = __capacity(data['highway'], lanes)


def __capacity(highway_val, lanes):
    """
    capacity estimation based on Zilske, Michael, Andreas Neumann, and Kai Nagel.
    OpenStreetMap for traffic simulation. Technische Universität Berlin, 2015.
    This certainly needs refinement and updating but shall suffice for now..
    """
    if lanes == 0: lanes = 1
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
        speed_list = [speed_mapping[item] for item in highway_val if speed_mapping[item] is not None]
        if len(speed_list) > 0:
            return min(speed_list)
        else:
            return default_speed


def save_pickle(g: nx.DiGraph, name: str):
    file_path = _filepath(name)
    nx.write_gpickle(g, file_path)
    log(f'saved to disk, {file_path=} ')


def load_pickle(name: str):
    file_path = _filepath(name, check_path_valid=True)
    return nx.read_gpickle(file_path)