#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
from dyntapy.network_data import get_from_ox_and_save, relabel_graph, save_pickle, load_pickle
from dyntapy.demand import add_centroids_to_graph, get_centroid_grid_coords
from dyntapy.settings import default_static_city as city
from dyntapy.visualization import show_network
from networkx import DiGraph


def get_graph(city=city, k=1, connector_type='link'):
    """

    Parameters
    ----------
    city : str
    k : int, connectors per centroid to be generated
    connector_type: ['turns' , 'links'] whether to add auto-configured link-connectors, k*2 for each centroid
    Returns
    -------
    DiGraph
    """
    g = get_from_ox_and_save(city, reload=False)
    x, y = get_centroid_grid_coords(city)
    g = add_centroids_to_graph(g, x, y, k=k, method=connector_type)
    g = relabel_graph(g)
    show_network(g)
    save_pickle(g, city + '_grid_centroids')
    return g


if __name__ == '__main__':
    get_graph()
