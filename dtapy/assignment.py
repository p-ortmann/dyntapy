#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from scipy.sparse import csr_matrix
from itertools import count
import networkx as nx
from scipy.sparse import lil_matrix
from stapy.demand import build_demand_structs
from numba.typed import List,Dict
from numba.core.types import int64, float64, UniTuple

class Assignment:
    """This class has no value when instantiated on its own,
     it merely sets up the state variables/interfaces to networkx"""

    def __init__(self, g: nx.DiGraph, od_matrix):
        """

        Parameters
        ----------
        g : nx.DiGraph
        od_matrix : array like object
            Dimensions should be nodes x nodes of the nx.DiGraph in the Assignment object
        """
        self._link_matrix=self.build_link_matrix()
        self.network= self.build_network()
        self.demand= self.build_demand()
        self.node_label, self.link_label= None, None,
        self.g = g
        # dict of dict with outer dict keyed by edges (u,v) and inner dict as data
        # to be dumped into the nx.DiGraph as key value pairs
        self.transform_graph_data()
        self.set_od_matrix(od_matrix)
        self.set_internal_labels(self.g, self.node_label, self.link_label)
    @staticmethod
    def build_network(g:nx.DiGraph):
        return 0
    @staticmethod
    def build_demand(od_matrix, g:nx.DiGraph):
        return 0
    @staticmethod
    def set_internal_labels(g:nx.DiGraph, node_labels, link_labels):
        # node- and link labels are both arrays in which the indexes refer to the internal IDs and the values to the
        #IDs used in nx
        node_labels=np.empty(g.number_of_nodes(), dtype=np.int64)
        link_labels=np.empty((g.number_of_edges(),2), dtype=np.int64)
        counter = count()
        for node_id, u in enumerate(g.nodes):
            g.nodes[u]['_id'] = node_id
            node_labels[node_id]=u
            for v in g.succ[u]:
                link_id = next(counter)
                g[u][v]['_id'] = link_id
                link_labels[link_id][0],link_labels[link_id][1]=u,v

    def build_link_matrix(self):
        #creates a 3D matrix that contains the full state on each link for all time periods,
        # see Links object for details
        _link_matrix = np.empty(shape=(g.number_of_edges, 4 + 2 * number_of_timesteps)
        return _link_matrix

    def transform_graph_data(self):
        """
        routine to consolidate existing link_ids and labelling logic
        Returns
        -------
        adjacency stores node and link ids in assignment reference e.g. adjacency[link_id](node_id,node_id) with
        labelling starting at 0
        translation_link_ids_nx stores the same information in reference of the underlying nx graph,
        e.g. translation_link_ids_nx[link_id]=(u,v) with u and v being node indices for self.g
        note that 'link_id' always refers to the assignment ids as the nx dicts references nodes by key (u,v)
        """
        self.adj_edge_list = List()
        for i in range(self.edge_order): self.adj_edge_list.append((0, 0))
        self.edge_map, self.inverse_edge_map = Dict(), Dict()
        self.translation_link_ids_nx = [None for _ in range(self.edge_order)]
        self.node_map_to_nx = [None for _ in range(self.node_order)]
        self.link_flows, self.link_capacities, self.link_ff_times, self.link_travel_times = \
            (np.zeros(self.g.number_of_edges()) for _ in range(4))
        counter = count()
        for node_id, u in enumerate(self.g.nodes):
            self.g.nodes[u]['_id'] = node_id
            self.node_map_to_nx[node_id] = u
            for v in self.g.succ[u]:
                link_id = next(counter)
                self.g[u][v]['_id'] = link_id
                self.translation_link_ids_nx[link_id] = (u, v)
                self.link_capacities[link_id] = self.g[u][v]['capacity']
                self.link_ff_times[link_id] = self.g[u][v]['travel_time']
        for u, v, link_id in self.g.edges.data('_id'):
            _u, _v = self.g.nodes[u]['_id'], self.g.nodes[v]['_id']
            self.adj_edge_list[link_id] = _u, _v
            self.edge_map[(_u, _v)] = link_id
            self.inverse_edge_map[link_id] = (_u, _v)

    def set_od_matrix(self, od_matrix):
        """
        sets OD matrix for assignment object, calculates production and attraction of nodes
        and writes results into node_data (to be written back)
        Parameters
        ----------
        od_matrix : array like object
            Dimensions should be nodes x nodes of the nx.DiGraph in the Assignment object

        Returns
        -------

        """
        assert isinstance(od_matrix, lil_matrix)
        assert od_matrix.sum() > 0
        assert od_matrix.shape == (self.g.number_of_nodes(), self.g.number_of_nodes())
        self.demand_dict, self.od_flow_vector = build_demand_structs(od_matrix)
        self.od_matrix = od_matrix.tocsr(copy=True)
        self.sparse_od_matrix = csr_matrix(self.od_matrix)
        originating_traffic = self.sparse_od_matrix.sum(axis=1)  # summing all rows
        destination_traffic = self.sparse_od_matrix.sum(axis=0).transpose()  # summing all columns
        for i, d in enumerate(originating_traffic):
            d = float(d)
            if d > 0:
                self.node_data[i]['originating_traffic'] = float(d)
        for i, d in enumerate(destination_traffic):
            d = float(d)
            if d > 0:
                self.node_data[i]['destination_traffic'] = float(d)







#my_array = np.arange(50000).reshape(10000, 5)
#my_network = Network(link_matrix=my_array)
#print(my_network.links.capacity[5])

# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise staionary characteristics
# example ramp metering event capacity choke, relative and absolute events



#TODO: jeroen which attributes are dynamic
#TODO: which ordering to use instar, outstar - ideally use both depending on your lagorithm yes both
#TODO: how high a priority is visualization.. - one time, interval visualization
# arrays for different things separately by node id
# what does the node data structure look like, turning fraction matrix
#data structure for events a sparse matrix .. time steps( row) * links(columns) value as capacity,
# different ones for each theme

#is the results object a good decision?, we can just pass on network/ demand objects for warm starting
#to enable syntax like DynamicAssignment.Network.Flows
# time incoming outgoing link, belgium network


