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
from network_data import get_from_ox_and_save, relabel_graph
from demand import add_centroids_to_graph, get_centroid_grid_coords
from assignment import Assignment

(g, deleted) = get_from_ox_and_save('Gent', reload=False)
x, y = get_centroid_grid_coords('Gent')

# need to define the number of centroids and connectors first as they are the first 'nodes' and 'links' of our network
# that needs to be accounted for in the labeling

tot_centroids = x.size
tot_connectors = tot_centroids * 3
g = relabel_graph(g, tot_centroids, tot_connectors)
print('relabeling passed')
add_centroids_to_graph(g, x, y, k=1)
print(f'number of nodes{g.number_of_nodes()}')
# plot_network(deleted, mode='deleted elements')
obj = Assignment(g, od_matrix=None)
