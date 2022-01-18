#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import os

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx.distance import euclidean_dist_vec

from dyntapy.supply_data import relabel_graph

default_capacity = 2000
default_free_speed = 80
default_lanes = 1
facility_type = "highway"
bottleneck_capacity = 500
bottleneck_free_speed = 120
default_node_ctrl_type = "none"


def get_toy_network(name, relabel=False):
    """
    creates toy network based and returns the corresponding GMNS conform MultiDiGraph
    Parameters
    ----------
    relabel : bool, whether to add link and node ids, only applicable if no centroids are needed.
    name : str, name of the toy network to get, see below

    Returns
    -------

    """
    g = nx.MultiDiGraph()
    if name == "cascetta":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": np.sqrt(2)}),
            (2, {"x_coord": np.sqrt(2) + 2, "y_coord": 2 * np.sqrt(2)}),
            (3, {"x_coord": np.sqrt(2) + 2, "y_coord": 0}),
            (4, {"x_coord": 2 * np.sqrt(2) + 2, "y_coord": np.sqrt(2)}),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        ebunch_of_edges = [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 3),
            (4, 2),
            (3, 2),
            (3, 1),
            (2, 1),
        ]
        bottle_neck_edges = [(2, 3), (3, 2), (3, 4), (4, 3)]
        g.add_edges_from(ebunch_of_edges)
        set_toy_network_attributes(g, bottle_neck_edges)

    elif name == "simple_bottleneck":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 1}),
            (2, {"x_coord": 3, "y_coord": 1}),
            (3, {"x_coord": 4, "y_coord": 1}),
            (4, {"x_coord": 5, "y_coord": 1}),
        ]
        ebunch_of_edges = [(2, 3), (3, 2), (1, 2), (2, 1), (3, 4), (4, 3)]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(2, 3), (3, 2)]
        set_toy_network_attributes(g, bottleneck_edges)
    elif name == "simple_merge":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 1}),
            (2, {"x_coord": 2, "y_coord": 2}),
            (3, {"x_coord": 4, "y_coord": 1.5}),
            (4, {"x_coord": 5, "y_coord": 1.5}),
            (5, {"x_coord": 6, "y_coord": 1.5}),
        ]
        ebunch_of_edges = [
            (2, 3),
            (3, 2),
            (1, 3),
            (3, 1),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(3, 4), (4, 3)]
        set_toy_network_attributes(g, bottleneck_edges)
    elif name == "simple_diverge":
        ebunch_of_nodes = [
            (1, {"x_coord": 2, "y_coord": 2}),
            (2, {"x_coord": 3, "y_coord": 2}),
            (3, {"x_coord": 4, "y_coord": 2}),
            (4, {"x_coord": 5, "y_coord": 1}),
            (5, {"x_coord": 5, "y_coord": 3}),
        ]
        ebunch_of_edges = [
            (2, 3),
            (3, 2),
            (1, 2),
            (2, 1),
            (3, 4),
            (4, 3),
            (3, 5),
            (5, 3),
        ]
        g.add_nodes_from(ebunch_of_nodes)
        g.add_edges_from(ebunch_of_edges)
        bottleneck_edges = [(2, 3), (3, 2)]
        set_toy_network_attributes(g, bottleneck_edges)
    elif name in [
        "chicagosketch",
        "chicagoregional",
        "philadelphia",
        "siouxfalls",
        "sydney",
        "birmingham",
    ]:
        # The source of these networks is Ben Stabler et al.,
        # see here: https://github.com/bstabler/TransportationNetworks
        # don't forget to cite as shown in the repository

        edge_file = "{0}{1}{2}_net.tntp".format(
            os.path.dirname(os.path.realpath(__file__)), os.path.sep, name
        )
        node_file = "{0}{1}{2}_node.tntp".format(
            os.path.dirname(os.path.realpath(__file__)), os.path.sep, name
        )
        edge_df = pd.read_csv(edge_file, skiprows=8, sep="\t")
        if name == "philadelphia":
            sep = " "
        elif name == "birmingham":
            sep = "       "
        else:
            sep = "\t"
        node_df = pd.read_csv(node_file, sep=sep)
        Graphtype = nx.MultiDiGraph()
        edge_df["init_node"] = edge_df["init_node"] - 1
        edge_df["term_node"] = edge_df["term_node"] - 1
        edge_df["free_speed"] = edge_df["length"] / edge_df["free_flow_time"]
        edge_df["lanes"] = 1
        g = nx.from_pandas_edgelist(
            edge_df,
            source="init_node",
            target="term_node",
            edge_attr=["length", "capacity", "free_speed", "lanes"],
            create_using=Graphtype,
        )
        node_df = node_df.rename(columns={col: col.lower() for col in node_df.columns})
        node_df["node"] = node_df["node"] - 1
        for node, x, y in zip(node_df["node"], node_df["x"], node_df["y"]):
            try:
                g.nodes[node]["x_coord"] = x
                g.nodes[node]["y_coord"] = y
            except KeyError:
                #   no edge with this node was added..
                continue
        g.graph = {"name": name}
        return relabel_graph(g)

    else:
        raise ValueError("no toy network provided under that name")
    g.graph = {"name": name}
    if not relabel:
        return g
    else:
        return relabel_graph(g)


def set_toy_network_attributes(g, bottleneck_edges):
    for v in g.nodes:
        g.nodes[v]["ctrl_type"] = default_node_ctrl_type
    for u, v, data in g.edges.data():
        y1 = g.nodes[v]["y_coord"]
        x1 = g.nodes[v]["x_coord"]
        y0 = g.nodes[u]["y_coord"]
        x0 = g.nodes[u]["x_coord"]
        data["length"] = euclidean_dist_vec(y0, x0, y1, x1)
        data["capacity"] = default_capacity
        data["free_speed"] = default_free_speed
        data["lanes"] = default_lanes
        if (u, v) in bottleneck_edges:
            data["capacity"] = bottleneck_capacity
            data["free_speed"] = bottleneck_free_speed


if __name__ == "__main__":
    get_toy_network("cascetta")
    for name in [
        "chicagosketch",
        "chicagoregional",
        "philadelphia",
        "siouxfalls",
        "sydney",
        "birmingham",
    ]:
        g = get_toy_network(name)
        print(f"got {name}")
