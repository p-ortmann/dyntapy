#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#

import networkx as nx
from settings import data_folder, visualization_keys_nodes, visualization_keys_edges
from geojson import LineString, Point, Feature, FeatureCollection, dumps,dump
import json
import os
import numpy as np


def nx_to_geojson(g: nx.DiGraph, to_file=False, file_name=None):
    '''
    g: networkx Digraph with geographic information as line strings (WGS84, 4326 unprojected) osm ids ..
    '''
    my_features = []
    for u, v, data in g.edges.data():
        if 'geometry' in data:
            zip(data['geometry'].xy[0], data['geometry'].xy[1])
            my_linestring = LineString([[x, y] for x,y in zip(data['geometry'].xy[0], data['geometry'].xy[1])])
            print(my_linestring)
        else:
            my_linestring = LineString([[g.nodes[u]['x'], g.nodes[u]['y']], [g.nodes[v]['x'], g.nodes[v]['y']]])
        tmp = {i: data[i] for i in data if i in visualization_keys_edges }
        tmp['time'], tmp['flow']= flow_to_time_profile(tmp['flow'])
        my_features.append(Feature(geometry=my_linestring, properties=tmp))
    for u, data in g.nodes.data():
        my_point = Point([data['x'], data['y']])
        tmp = {i: data[i] for i in data if i in visualization_keys_nodes}
        my_features.append(Feature(geometry=my_point, properties=tmp))
    if to_file:
        if file_name is None:
            try:
                file_name = 'geojson results ' + g.graph['name']
            except KeyError:
                # no name provided ..
                file_name = 'geojson results ' + '... provide city name in graph and it will show here..'
        text_file = open(os.path.join(os.getcwd(), os.path.join(data_folder, file_name)), "w", )
        dump(FeatureCollection(my_features), fp=text_file, cls=NpEncoder)
        text_file.close()
        print('json dumped to file')
    else:
        return dumps(FeatureCollection(my_features))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
def flow_to_time_profile(val):
    time =[360*60+i*15*60 for i in np.arange(0,17)]
    middle=round(len(time)/2)
    factors=np.arange(0.0,1.0,1.0/middle)
    factors=np.concatenate((np.append(factors,1), np.flip(factors)), axis=None)
    flows=[val*factor for factor in factors]
    return time, flows


