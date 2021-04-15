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
from dtapy.network_data import get_from_ox_and_save, relabel_graph, save_pickle, load_pickle, sort_graph
from dtapy.demand import add_centroids_to_graph, get_centroid_grid_coords
from dtapy.settings import parameters
from dtapy.settings import default_city as city
from dtapy.visualization import show_network
from dtapy.utilities import log, _filepath
default_centroid_spacing = parameters.demand.default_centroid_spacing

g = get_from_ox_and_save(city, reload=False)
x, y = get_centroid_grid_coords(city)

k = 2  # connectors per centroid to be generated
g = add_centroids_to_graph(g, x, y, k=k)
g = relabel_graph(g)
show_network(g)
save_pickle(g, city + '_grid_centroids')
