#    This file is part of the Traffic Assignment Package developed at KU Leuven.
#   Copyright (c) 2020 Paul Ortmann
#   License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import os
import networkx as nx
import numpy as np
import osmnx as ox
from stapy.settings import config_dict, speed_mapping, cap_mapping, default_capacity, default_speed
from stapy import data_folder
from stapy.utilities import log
traffic_keys = config_dict['traffic_keys']
def create_cascetta_nw():
    ox.graph_from_place()
    """creates toy network based on cascetta and returns the corresponding TrafficGraph"""
    g = nx.DiGraph()
    ebunch_of_nodes = [
        (1, {'x': 0, 'y': np.sqrt(2)}),
        (2, {'x': np.sqrt(2), 'y': 2 * np.sqrt(2)}),
        (3, {'x': np.sqrt(2), 'y': 0}),
        (4, {'x': 2 * np.sqrt(2), 'y': np.sqrt(2)})]
    ebunch_of_edges = [
        (1, 2, {'length':2000}), (1, 3, {'length':2000}), (2, 3, {'length':2000 * np.sqrt(2)}), (2, 4, {'length':2000}),
        (3, 4, {'length':2000}), (4, 3, {'length':2000}), (4, 2, {'length':2000}), (3, 2, {'length':2000 * np.sqrt(2)}),
        (3, 1, {'length':2000}), (2, 1, {'length':2000})]
    g.add_nodes_from(ebunch_of_nodes)
    g.add_edges_from(ebunch_of_edges)
    for (u, v) in g.edges():
        g[u][v]['capacity'] = np.int(2000)
        g[u][v]['maxspeed'] = np.int(120)
        if (u, v) in [(2, 3), (3, 2)]:
            g[u][v]['capacity'] = np.int(2000)
            g[u][v]['maxspeed'] = np.int(80)
    set_free_flow_travel_times(g)
    return g


def get_from_ox_and_save(name: str, plot_deleted=True, reload=False):
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
    set_free_flow_travel_times(dir_g)
    nx.write_gpickle(dir_g, file_path)
    assert 'crs' in dir_g.graph
    dir_g.graph['name'] = name
    deleted.graph['name'] = name
    return dir_g, deleted


def set_free_flow_travel_times(g: nx.DiGraph):
    """
    free flow travel times in minutes stored as 'travel_time'

    Returns
    -------

    """
    for u, v, data in g.edges.data():
        try:
            if type(data['maxspeed']) == list:
                speed = max(data['maxspeed'])
            elif isinstance(data['maxspeed'], str):
                # print(f' u: {u} v: {v} and data: {data}')
                speed = int(data['maxspeed'])
            else:
                speed = data['maxspeed']
            if isinstance(data['length'], str):
                # print(f' u: {u} v: {v} and data: {data}')
                data['length'] = int(data['length'])
            g[u][v]['travel_time'] = np.float(data['length'] / (speed/3.6))
            #print(g[u][v]['travel_time'])
        except KeyError:
            print(f"insufficient data for edge {u} {v} data: {data};length not provided..")


def __clean_up_data(g: nx.DiGraph):
    for u, v, data in g.edges.data():
        tmp = set(data.keys()).intersection(traffic_keys)  # filter keys for all traffic related (numerical) attr
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
                print(data)
                data['maxspeed'] = 50

        try:
            lanes = int(data['lanes'])
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, TypeError):
                assert isinstance(data['lanes'], list)
                try:
                    lanes = min([int(val) for val in data['lanes']])
                except ValueError:
                    lanes=1
            else:
                lanes = 1
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