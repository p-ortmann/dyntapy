#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import osmnx as ox
import networkx as nx
import numpy as np
from osmnx.distance import euclidean_dist_vec
from dtapy.network_data import relabel_graph

default_capacity = 2000
default_free_speed = 80
default_lanes = 1
facility_type = 'highway'
bottleneck_capacity = 500
bottleneck_free_speed = 120
default_node_ctrl_type = 'none'


def get_toy_network(name='cascetta', relabel=False):
    """
    creates toy network based and returns the corresponding GMNS conform MultiDiGraph
    Parameters
    ----------
    relabel : bool, whether to add link and node ids, only applicable if no centroids are needed.
    name : str, name of the toy network to get, currently supported: 'cascetta'

    Returns
    -------

    """
    g = nx.MultiDiGraph()
    if name == 'cascetta':
        ebunch_of_nodes = [
            (1, {'x_coord': 2, 'y_coord': np.sqrt(2)}),
            (2, {'x_coord': np.sqrt(2) + 2, 'y_coord': 2 * np.sqrt(2)}),
            (3, {'x_coord': np.sqrt(2) + 2, 'y_coord': 0}),
            (4, {'x_coord': 2 * np.sqrt(2) + 2, 'y_coord': np.sqrt(2)})]
        g.add_nodes_from(ebunch_of_nodes)
        ebunch_of_edges = [
            (1, 2), (1, 3), (2, 3), (2, 4),
            (3, 4), (4, 3), (4, 2), (3, 2),
            (3, 1), (2, 1)]
        bottle_neck_edges = [(2, 3), (3, 2)]
        g.add_edges_from(ebunch_of_edges)
        set_toy_network_attributes(g, bottle_neck_edges)

    elif name == 'simple_bottleneck':
        ebunch_of_nodes = [
            (1, {'x_coord': 2, 'y_coord': 1}),
            (2, {'x_coord': 3, 'y_coord': 1}),
            (3, {'x_coord': 4, 'y_coord': 1}),
            (4, {'x_coord': 5, 'y_coord': 1})]
        ebunch_of_edges = [(2, 3), (3, 2), (1, 2), (2, 1),
                           (3, 4), (4, 3)]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(2, 3), (3, 2)]
        set_toy_network_attributes(g, bottleneck_edges)
    elif name == 'simple_merge':
        ebunch_of_nodes = [
            (1, {'x_coord': 2, 'y_coord': 1}),
            (2, {'x_coord': 2, 'y_coord': 2}),
            (3, {'x_coord': 4, 'y_coord': 1.5}),
            (4, {'x_coord': 5, 'y_coord': 1.5}),
            (5, {'x_coord': 6, 'y_coord': 1.5})]
        ebunch_of_edges = [(2, 3), (3, 2), (1, 3), (3, 1),
                           (3, 4), (4, 3), (4, 5), (5, 4)]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(3, 4), (4,3)]
        set_toy_network_attributes(g, bottleneck_edges)
    elif name == 'simple_diverge':
        ebunch_of_nodes = [
            (1, {'x_coord': 2, 'y_coord': 2}),
            (2, {'x_coord': 3, 'y_coord': 2}),
            (3, {'x_coord': 4, 'y_coord': 1}),
            (4, {'x_coord': 4, 'y_coord': 3})]
        ebunch_of_edges = [(2, 3), (3, 2), (1, 2), (2, 1),
                           (2, 4), (4, 2)]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = []
        set_toy_network_attributes(g, bottleneck_edges)
    else:
        raise ValueError('no toy network provided under that name')
    if not relabel:
        return g
    else:
        return relabel_graph(g)


def set_toy_network_attributes(g, bottleneck_edges):
    for v in g.nodes:
        g.nodes[v]['ctrl_type'] = default_node_ctrl_type
    for u, v,data in g.edges.data():
        y1 = g.nodes[v]['y_coord']
        x1 = g.nodes[v]['x_coord']
        y0 = g.nodes[u]['y_coord']
        x0 = g.nodes[u]['x_coord']
        data['length'] = euclidean_dist_vec(y0, x0, y1, x1)
        data['capacity'] = default_capacity
        data['free_speed'] = default_free_speed
        data['lanes'] = default_lanes
        if (u, v) in bottleneck_edges:
            data['capacity'] = bottleneck_capacity
            data['free_speed'] = bottleneck_free_speed
