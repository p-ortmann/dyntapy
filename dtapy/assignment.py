#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from scipy.sparse import csr_matrix
from itertools import count
import networkx as nx
from scipy.sparse import lil_matrix
from stapy.demand import build_demand_structs
from numba.typed import List, Dict
from numba.core.types import uint32, float32, int32
from datastructures.csr import csr_prep, UI32CSRMatrix, F32CSRMatrix
from dtapy.core.jitclasses import Links, Nodes, Network, Turns, Demand
from dtapy.parameters import v_wave_default, turn_capacity_default, turn_type_default, node_capacity_default, \
    node_control_default, turn_t0_default


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
        self.g = g
        self.node_label, self.link_label, self.node_adjacency = self.set_internal_labels(self.g)
        self.number_of_nodes = np.uint32(self.g.number_of_nodes())
        self.number_of_links = np.uint32(self.g.number_of_edges())
        self.number_of_turns = None
        self.number_of_time_steps = np.uint32(10)
        # self.demand = self.build_demand()
        self.network = self.build_network()
        print('network build, lets see')
        print()
        # dict of dict with outer dict keyed by edges (u,v) and inner dict as data
        # to be dumped into the nx.DiGraph as key value pairs
        # self.set_od_matrix(od_matrix)

    def build_demand(self, od_matrix, g: nx.DiGraph):
        origins, destinations, number_of_timesteps = None, None, None
        return Demand(od_matrix, origins, destinations, number_of_timesteps)

    def build_network(self):
        nodes = self.__build_nodes()
        turns = self.__build_turns(nodes)
        links = self.__build_links(turns)
        return Network(links, nodes, turns, self.number_of_nodes, self.number_of_links, self.number_of_turns)

    def __build_nodes(self):
        link_ids = np.arange(self.g.number_of_edges(), dtype=np.uint32)
        values, col, row = csr_prep(self.node_adjacency, link_ids, (self.number_of_nodes, self.number_of_nodes))
        forward = UI32CSRMatrix(values, col, row)
        values, col, row = csr_prep(np.column_stack((self.node_adjacency[:, 1], self.node_adjacency[:, 0])),
                                    link_ids, (self.number_of_nodes, self.number_of_nodes))
        backward = UI32CSRMatrix(values, col, row)
        capacity = np.full(self.number_of_nodes, node_capacity_default, dtype=np.float32)
        control_type = np.full(self.number_of_nodes, node_control_default, dtype=np.int8)
        return Nodes(forward, backward, control_type, capacity)

    def __build_turns(self, nodes: Nodes):
        to_nodes = List()
        from_nodes = List()
        from_links = List()
        to_links = List()
        turn_id = List()
        turn_counter = 0
        for via_node in np.arange(self.number_of_nodes):
            # named here _attribute to indicate all the to nodes/links that are associated with the via_node
            _to_nodes = nodes.forward.get_nnz(via_node)
            _from_nodes = nodes.backward.get_nnz(via_node)
            _from_links = nodes.backward.get_row(via_node)
            _to_links = nodes.forward.get_row(via_node)
            for to_node, from_node, from_link, to_link in zip(_to_nodes, _from_nodes, _from_links, _to_links):
                turn_id.append(turn_counter)
                to_nodes.append(to_node)
                from_nodes.append(from_node)
                from_links.append(from_link)
                to_links.append(to_link)
                turn_counter += 1
        number_of_turns = turn_counter + 1
        self.number_of_turns = number_of_turns
        capacity = np.full(number_of_turns, turn_capacity_default, dtype=np.float32)
        turn_type = np.full(number_of_turns, turn_type_default, dtype=np.int8)
        t0 = np.full(number_of_turns, turn_t0_default, dtype=np.float32)
        return Turns(t0, capacity, np.array(from_nodes, dtype=np.uint32), np.arange(self.number_of_nodes, dtype=np.uint32),
              np.array(to_nodes, dtype=np.uint32), np.array(from_links, dtype=np.uint32),
              np.array(to_links, dtype=np.uint32), turn_type)

    def __build_links(self, turns):
        """
        initiates all the different numpy arrays for the links object from nx.DiGraph,
        requires the networkx graph to be set up as specified in the network_data
        Returns
        links : Links
        -------

        """
        length = np.empty(self.g.number_of_edges(), dtype=np.float32)
        capacity = np.empty(self.g.number_of_edges(), dtype=np.float32)
        v0_prt = np.empty(self.g.number_of_edges(), dtype=np.float32)
        lanes = np.empty(self.g.number_of_edges(), dtype=np.uint8)
        from_node = self.node_adjacency[:, 0]
        to_node = self.node_adjacency[:, 1]
        link_ids = np.arange(self.g.number_of_edges())
        for _id, arr in zip(link_ids, self.link_label):
            u, v = arr
            print(f'{u=}{v=}{_id=}')
            length[_id] = self.g[u][v]['length']
            capacity[_id] = self.g[u][v]['capacity']
            v0_prt[_id] = self.g[u][v]['maxspeed']
            lanes[_id] = self.g[u][v]['lanes']
        costs = np.empty((self.number_of_time_steps, self.number_of_links), dtype=np.float32)
        cvn_up = np.empty((self.number_of_time_steps, self.number_of_links), dtype=np.float32)
        cvn_down = np.empty((self.number_of_time_steps, self.number_of_links), dtype=np.float32)
        v_wave = np.full(self.number_of_links, v_wave_default, dtype=np.float32)
        number_of_turns = np.uint32(len(turns.to_link))
        fw_index_array = np.column_stack((turns.from_link, turns.to_link))
        bw_index_array = np.column_stack((turns.to_link, turns.from_link))
        val = np.arange(number_of_turns, dtype=np.uint32)
        val, col, row = csr_prep(fw_index_array, val, self.number_of_links)
        forward = UI32CSRMatrix(val, col, row)
        val, col, row = csr_prep(bw_index_array, val, self.number_of_links)
        backward = UI32CSRMatrix(val, col, row)

        return Links(length, from_node, to_node, capacity, v_wave, costs,v0_prt, cvn_up, cvn_down, forward, backward,lanes)

    @staticmethod
    def set_internal_labels(g: nx.DiGraph):
        # node- and link labels are both arrays in which the indexes refer to the internal IDs and the values to the
        # IDs used in nx
        node_labels = np.empty(g.number_of_nodes(), np.uint64)
        link_labels = np.empty((g.number_of_edges(), 2), np.uint64)
        counter = count()
        for node_id, u in enumerate(g.nodes):
            g.nodes[u]['_id'] = node_id
            node_labels[node_id] = u
            for v in g.succ[u]:
                link_id = next(counter)
                g[u][v]['_id'] = link_id
                link_labels[link_id][0], link_labels[link_id][1] = u, v
        node_adjacency = np.empty((g.number_of_edges(), 2), dtype=np.uint32)
        for u, v, data in g.edges.data():

            _id = data['_id']
            node1 = g.nodes[u]['_id']
            node2 = g.nodes[v]['_id']
            node_adjacency[_id] = node1, node2

        return node_labels, link_labels, node_adjacency

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

# my_array = np.arange(50000).reshape(10000, 5)
# my_network = Network(link_matrix=my_array)
# print(my_network.links.capacity[5])

# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise staionary characteristics
# example ramp metering event capacity choke, relative and absolute events


# TODO: jeroen which attributes are dynamic
# TODO: which ordering to use instar, outstar - ideally use both depending on your lagorithm yes both
# TODO: how high a priority is visualization.. - one time, interval visualization
# arrays for different things separately by node id
# what does the node data structure look like, turning fraction matrix
# data structure for events a sparse matrix .. time steps( row) * links(columns) value as capacity,
# different ones for each theme

# is the results object a good decision?, we can just pass on network/ demand objects for warm starting
# to enable syntax like DynamicAssignment.Network.Flows
# time incoming outgoing link, belgium netwo
