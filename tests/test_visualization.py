import networkx as nx
import numpy as np

from dyntapy import show_network, add_centroids, relabel_graph, show_demand, \
    add_connectors
from pytest import mark
from dyntapy.demand_data import od_graph_from_matrix
from dyntapy.supply_data import _set_toy_network_attributes, build_network
from dyntapy.settings import parameters
from dyntapy.demand import build_internal_static_demand, \
    build_internal_dynamic_demand, DynamicDemand, SimulationTime


@mark.skip(reason='only for visual inspection')
def test_visualization():
    g = nx.DiGraph()
    ebunch_of_nodes = [
        (1, {"x_coord": 12, "y_coord": 20}),
        (2, {"x_coord": 20, "y_coord": 20}),
        (3, {"x_coord": 6, "y_coord": 14}),
        (4, {"x_coord": 26, "y_coord": 14}),
        (5, {"x_coord": 16 , "y_coord": 3}),
    ]
    g.add_nodes_from(ebunch_of_nodes)
    ebunch_of_edges = [
        (1, 2),
        (2, 1),
        (4, 2),
        (2, 4),
        (1, 3),
        (3, 1),
        (3, 5),
        (5, 3),
        (4, 5),
        (5, 4),
    ]
    bottle_neck_edges = [(2, 4), (4, 2)]
    g.add_edges_from(ebunch_of_edges)
    _set_toy_network_attributes(g, bottle_neck_edges)
    # show_network(g, euclidean=True)
    if False:
        # making crossing links short to have routes competitive
        g[2][3]['length'] = 1
        g[3][2]['length'] = 1
        g[5][4]['length'] = 1
        g[4][5]['length'] = 1
    centroid_x = np.array([12, 20, 16])
    centroid_y = np.array([18, 18, 1])
    g = add_centroids(g, centroid_x, centroid_y, k=1, method='turn', euclidean=True)
    # also adds connectors automatically
    g = relabel_graph(g)  # adding link and node ids, connectors and centroids
    # are the first elements
    show_network(g, euclidean=True)
