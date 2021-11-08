#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import networkx as nx
from dyntapy.sta.algorithms.graph_utils import make_out_links, make_in_links
from dyntapy.utilities import log
from dyntapy.sta.demand import build_static_demand
import numpy as np
from numba.typed import List, Dict
from warnings import warn


class StaticAssignment:
    def __init__(self, g: nx.DiGraph, od_graph: nx.DiGraph):
        """

        Parameters
        ----------
        g : nx.DiGraph
        od_matrix : array like object
            Dimensions should be nodes x nodes of the nx.DiGraph in the Assignment object
        """
        self.adj_edge_list, self.link_flows, \
        self.link_ff_times, self.link_travel_times, self.link_capacities, self.destinations, \
        self.demand_dict, self.edge_map, self.od_flow_vector, self.inverse_edge_map = None, None, None, None, None, None, \
                                                                                      None, None, None, None
        self.from_nodes, self.to_nodes = None, None
        self.g = g
        self.tot_connectors = None
        self.tot_nodes = self.g.number_of_nodes()
        self.tot_links = self.g.number_of_edges()
        # dict of dict with outer dict keyed by edges (u,v) and inner dict as data
        # to be dumped into the nx.DiGraph as key value pairs
        self.fill_graph_structures()
        self.out_links = make_out_links(adj_list=self.adj_edge_list, number_of_nodes=self.tot_nodes)
        self.in_links = make_in_links(adj_list=self.adj_edge_list, number_of_nodes=self.tot_nodes)
        self.od_graph = od_graph
        self.od_matrix = nx.to_scipy_sparse_matrix(od_graph, weight='flow', format='lil')
        self.demand = build_static_demand(od_graph)
        log('Assignment object initialized!')
        print('init passed successfully')

    def fill_graph_structures(self):
        """
        a bit of a mess, somewhat copied from the dynamic implementation,
        works tho
        Returns
        -------
        adjacency stores node and link ids in assignment reference e.g. adjacency[link_id](node_id,node_id) with
        labelling starting at 0
        """
        g = self.g
        edge_data = [(_, _, data) for _, _, data in g.edges.data()]
        sorted_edges = sorted(edge_data, key=lambda t: t[2]['link_id'])
        sorted_nodes = sorted(g.nodes(data=True), key=lambda t: t[1]['node_id'])
        node_ids = np.array([data['node_id'] for (_, data) in sorted_nodes], dtype=np.uint32)
        if not np.all(node_ids[1:] == node_ids[:-1] + 1):
            raise ValueError('the node_ids in the graph are assumed to be monotonously increasing and have to be '
                             'added accordingly')
        self.from_nodes = np.array([d['from_node_id'] for (_, _, d) in sorted_edges], dtype=np.uint32)
        self.to_nodes = np.array([d['to_node_id'] for _, _, d in sorted_edges], dtype=np.uint32)
        # i'll leave these variables here for now, could be useful for the future
        link_ids = np.array([d['link_id'] for _, _, d in sorted_edges], dtype=np.uint32)
        if not np.all(link_ids[1:] == link_ids[:-1] + 1):
            raise ValueError('the node_ids in the graph are assumed to be monotonously increasing and have to be '
                             'added accordingly')

        link_type = np.array([np.int8(d.get('link_type', 0)) for (_, _, d) in sorted_edges], dtype=np.int8)
        capacities = np.array([d['capacity'] for (_, _, d) in sorted_edges], dtype=np.float32)
        free_speed = np.array([d['free_speed'] for (_, _, d) in sorted_edges], dtype=np.float32)
        lanes = np.array([d['lanes'] for (_, _, d) in sorted_edges], dtype=np.uint8)
        if min(lanes) < 1:
            warn('some roads have zero lanes, minimum of one lane assumed')
        length = np.array([d['length'] for (_, _, d) in sorted_edges], dtype=np.float32)
        self.link_capacities = np.maximum(capacities, lanes * capacities)
        self.link_ff_times = length / free_speed
        max_length = np.max(length)
        if np.max(length) > 100:
            warn(f'Network contains very long links, up to {max_length} km. Implementation has not been verified for'
                 f'this type of network. calculations may yield unexpected results.')

        self.tot_connectors = np.argwhere(link_type == 1).size + np.argwhere(link_type == -1).size
        # 1 is for sources (connectors leading out of a centroid)
        # -1 for sinks (connectors leading towards a centroid)
        # static bit from here
        self.adj_edge_list = List()
        for i in range(self.tot_links):
            self.adj_edge_list.append((0, 0))
        self.edge_map, self.inverse_edge_map = Dict(), Dict()
        self.link_flows, self.link_travel_times = \
            (np.zeros(self.g.number_of_edges()) for _ in range(2))
        for link_id, (u, v) in enumerate(zip(self.from_nodes, self.to_nodes)):
            self.adj_edge_list[link_id] = u, v
            self.edge_map[(u, v)] = link_id
            self.inverse_edge_map[link_id] = (u, v)
