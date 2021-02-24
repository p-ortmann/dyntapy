#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import os
import networkx as nx
import numpy as np
import osmnx as ox
from settings import parameters
from __init__ import data_folder
from utilities import log
from itertools import count

speed_mapping = parameters.supply.speed_mapping
cap_mapping = parameters.supply.cap_mapping
default_capacity = parameters.supply.cap_mapping
default_speed = parameters.supply.default_speed





def get_from_ox_and_save(name: str, reload=False):
    file_path = __filepath(name)
    _ = file_path + '_pure_ox'
    if not reload:
        try:
            g = nx.read_gpickle(_)
        except FileNotFoundError:
            log(f'city {name} could not be found in data folder, loading from osm', level=50)
            g = ox.graph_from_place(name, network_type='drive')
            nx.write_gpickle(g, file_path + '_pure_ox')
    else:
        log('Starting to load from OSM')
        g = ox.graph_from_place(name, network_type='drive')
        log('Finished downloading')
        nx.write_gpickle(g, file_path + '_pure_ox')
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
    __clean_up_data(dir_g)
    convert_to_gmns(dir_g)
    nx.write_gpickle(dir_g, file_path)
    assert 'crs' in dir_g.graph
    dir_g.graph['name'] = name
    return dir_g, deleted
def convert_to_gmns(g):
    # the attribute names are chosen in compliance with GMNS,
    # see https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/node.schema.json
    # and https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/link.schema.json
    # potentially could be extended to handle time of day changes in the future, see
    # https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/link_tod.schema.json
    # and also lanes:
    # https://github.com/zephyr-data-specs/GMNS/blob/master/Specification/lane_tod.schema.json
    edge_keys = ['link_id', 'from_node_id', 'to_node_id', 'length', 'capacity', 'flow', 'cost', 'max_speed', 'osm_id',
                 'name', 'facility_type']
    node_keys = ['node_id', 'x_coord', 'y_coord', 'osm_id', 'node_type', 'ctrl_type']

    for node in g.nodes:
        data = g.nodes[node]
        new_data = {}
        new_data['node_id']=node
        new_data['x_coord'] =data['x']
        new_data['y_coord'] = data['y']
        new_data['node_type']=None
        new_data['ctrl_type']=None
        g.nodes[node] =new_data
    for u,v in g.edges:
        data = g[u][v]
        new_data = {}
        new_data['link_id'] = data['osm_id']
        new_data['from_node_id'] = u
        new_data['to_node_id']= v
        new_data['length'] = data['length']
        new_data['free_speed'] = data['max_speed']
        new_data['name'] = data['name']
        new_data['facility_type']= data['highway']
        new_data['lanes'] =data['lanes']
        g[u][v]=new_data
def relabel_graph(g, number_of_centroids, number_of_connectors):
    """
    osmnx labels the graph nodes and edges by their osm ids. These are neither stable nor continuous. We relabel nodes and edges
    with our internal ids.
    Parameters
    ----------
    number_of_connectors : number of connectors in the graph
    g : nx.DiGraph
    number_of_centroids: int
    the first nodes are centroids, this is to reserve spots for them
    Returns
    -------
    nx.Digraph with continuously labelled nodes, consistent with internal notation


    """

    new_g = nx.MultiDiGraph()
    new_g.graph = g.graph
    link_counter = count(number_of_connectors)
    ordered_nodes = g.nodes
    for node_id, u in enumerate(ordered_nodes):
        _id = node_id + number_of_centroids
        data = g.nodes[u]
        new_g.add_node(_id, **data)
        new_g.nodes[_id]['osm_id'] = u
        g.nodes[u]['_id'] = _id
    for start_node, u in enumerate(ordered_nodes):
        _start_node = start_node + number_of_centroids
        for v in g.succ[u]:
            link_id = next(link_counter)
            end_node = g.nodes[v]['node_id']
            data = g[u][v]
            data['_id'] = link_id
            new_g.add_edge(start_node, end_node, **data)
            g[u][v]['_id'] = link_id
        # Note that the out_links of a given node always have consecutive ids
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
                if key == 'lanes':
                    print(data[key])
                print()
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
                print(data)
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
        data['lanes'] = lanes
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

        return cap_mapping[highway_val] * lanes
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
    file_path = __filepath(name)
    nx.write_gpickle(g, file_path)


def load_pickle(name: str):
    file_path = __filepath(name, check_path_valid=True)
    return nx.read_gpickle(file_path)


def __filepath(name: str, check_path_valid=False):
    assert isinstance(name, str)
    file_path = os.path.join(data_folder, str(name.lower() + '.pickle'))
    if check_path_valid:
        if not os.path.isfile(file_path):
            print(f'{name}.pickle not found in data folder!')
            raise NameError.with_traceback()
    return file_path
