#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import networkx as nx
from dyntapy.settings import dynamic_parameters
from dyntapy.__init__ import results_folder
from geojson import LineString, Point, Feature, dumps, dump, GeoJSON
import json
import os
import numpy as np

visualization_keys_nodes = dynamic_parameters.visualization.node_keys
visualization_keys_edges = dynamic_parameters.visualization.link_keys


def nx_to_geojson(g: nx.DiGraph, meta_data=None, to_file=False, city_name=None, indent = 0):
    '''
    g: networkx Digraph with geographic information as line strings (WGS84, 4326 unprojected) osm ids ..
    '''
    my_link_features = []
    my_node_features = []
    for u, v, data in g.edges.data():
        if 'geometry' in data:
            zip(data['geometry'].xy[0], data['geometry'].xy[1])
            my_linestring = LineString([[x, y] for x, y in zip(data['geometry'].xy[0], data['geometry'].xy[1])])
        else:
            my_linestring = LineString(
                [[g.nodes[u]['x_coord'], g.nodes[u]['y_coord']], [g.nodes[v]['x_coord'], g.nodes[v]['y_coord']]])
        tmp = {i: _json_formatting(data[i]) for i in data if i in visualization_keys_edges}
        my_link_features.append(Feature(geometry=my_linestring, properties=tmp))
    for u, data in g.nodes.data():
        my_point = Point([data['x_coord'], data['y_coord']])
        tmp = {i: _json_formatting(data[i]) for i in data if i in visualization_keys_nodes}
        my_node_features.append(Feature(geometry=my_point, properties=tmp))
    if to_file:
        if city_name is None:
            try:
                city_name = g.graph['name']
            except KeyError:
                # no name provided ..
                city_name = '... provide city name in graph and it will show here..'
        link_file_name = f'links_{city_name}.json'
        node_file_name = f'nodes_{city_name}.json'
        for filename, features in zip([link_file_name, node_file_name], [my_link_features, my_node_features]):
            text_file = open(os.path.join(os.getcwd(), os.path.join(results_folder, filename)), "w", )
            dump(FeatureCollection(features, meta_data), fp=text_file, cls=NpEncoder, indent=indent)
            text_file.close()
            print(f'json dumped to file at {os.getcwd() + os.path.sep + os.path.join(results_folder, filename)}')
    else:
        return dumps(FeatureCollection(my_link_features, meta_data), cls=NpEncoder, indent=indent), \
               dumps(FeatureCollection(my_node_features, meta_data), cls=NpEncoder, indent=indent)
    


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)


def _json_formatting(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return round(obj,6)
    elif isinstance(obj, float):
        return round(obj,6)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


class FeatureCollection(GeoJSON):
    """
    Represents a FeatureCollection, a set of multiple Feature objects.
    """

    def __init__(self, features, meta_data, **extra):
        """
        Initialises a FeatureCollection object from the
        :param features: List of features to constitute the FeatureCollection.
        :type features: list
        :return: FeatureCollection object
        :rtype: FeatureCollection
        """
        super(FeatureCollection, self).__init__(**extra)
        self["meta_data"] = meta_data
        self["features"] = features

    def errors(self):
        return self.check_list_errors(lambda x: x.errors(), self.features)

    def __getitem__(self, key):
        try:
            return self.get("features", ())[key]
        except (KeyError, TypeError, IndexError):
            return super(GeoJSON, self).__getitem__(key)
