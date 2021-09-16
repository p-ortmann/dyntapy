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
from dyntapy.settings import dynamic_parameters
from dyntapy.settings import default_dynamic_city as city
from dyntapy.visualization import show_network


def get_graph(city=city):
    g = get_from_ox_and_save(city, reload=False)
    x, y = get_centroid_grid_coords(city)

    k = 1  # connectors per centroid to be generated, for now just tested for k==1 for static
    method = 'link'  # link connectors for static assignments, turn for dynamic
    g = add_centroids_to_graph(g, x, y, k=k, method=method)
    g = relabel_graph(g)
    show_network(g)
    save_pickle(g, city + '_grid_centroids')
    return g
if __name__=='__main__':
    get_graph()