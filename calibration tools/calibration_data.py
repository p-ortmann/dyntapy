#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import googlemaps
from datetime import datetime
from googlemaps.distance_matrix import distance_matrix
import networkx as nx
import pyproj


def get_all_locations(g:nx.DiGraph):
    '''
    gets all individual locations from g and transforms them to lat long (4326)

    Parameters
    ----------
    g : networkx Digraph with locations stored as x, y in nodes in webmercator (3857)

    Returns
    -------
    list of locations in lat long

    '''

    in_proj, out_proj = pyproj.Proj('epsg:3857'), pyproj.Proj('epsg:4326'),
    node_xs_3857=[x for _,x in g.nodes.data('x')]
    node_ys_3857=[y for _,y in g.nodes.data('y')]

    node_xs, node_ys = pyproj.transform(in_proj, out_proj, node_xs_3857, node_ys_3857)
    return [(x,y) for x,y in zip(node_xs,node_ys) ]

