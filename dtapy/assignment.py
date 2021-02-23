#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from itertools import count
import networkx as nx
from numba.typed import List
from datastructures.csr import csr_prep, UI32CSRMatrix
from dtapy.core.assignment_cls import Links, Nodes, Network, Turns, DynamicDemand, SimulationTime
from dtapy.demand import _build_demand, _check_centroid_connectivity
from dtapy.parameters import parameters
from dtapy.core.assignment_methods import i_ltm_aon
from dataclasses import dataclass

v_wave_default = parameters.supply.v_wave_default
turn_capacity_default = parameters.supply.turn_capacity_default
turn_type_default = parameters.supply.turn_type_default
node_capacity_default = parameters.supply.node_capacity_default
turn_t0_default = parameters.supply.turn_t0_default
node_control_default = parameters.supply.node_control_default
network_loading_method = parameters.network_loading.method


class Assignment:
    """This class has no value when instantiated on its own,
     it merely sets up the state variables/interfaces to networkx and the demand generation"""

    def __init__(self, g: nx.DiGraph, time=SimulationTime(0, 24, parameters.network_loading.time_step),
                 method=i_ltm_aon):
        """

        Parameters
        ----------
        g : nx.DiGraph
        time : SimulationTime object, specifies start and end of the simulation and the time step to be used
        """
        _check_centroid_connectivity(g)
        self.g = g
        self.time = self.__init_time_obj(time)
        self.tot_nodes = np.uint32(self.g.number_of_nodes())
        self.tot_links = np.uint32(self.g.number_of_edges())
        self.tot_turns = None
        self.results = None
        #get adjacency from nx, and
        self.number_of_time_steps = np.uint32(10)
        # self.demand = self.build_demand()
        self.network = self.__build_network()
        print('network build')
        self.dynamic_demand: DynamicDemand = self.__build_demand()
        print('demand simulation build')
        self.network_loading_data_structs(method=network_loading_method)
        print('DNL data structures build')
        # replacing network object with specialized version for network loading method
        print('network build, lets see')
        # dict of dict with outer dict keyed by edges (u,v) and inner dict as data
        # to be dumped into the nx.DiGraph as key value pairs
        # self.set_od_matrix(od_matrix)

    def network_loading_data_structs(self, method: str):
        if method == 'iltm':
            from dtapy.core.network_loading.link_models.i_ltm_setup import i_ltm_setup
            i_ltm_setup(self)

    def __build_network(self):
        nodes = self.__build_nodes()
        print("nodes passed")
        turns = self.__build_turns(nodes)
        print("turns passed")
        links = self.__build_links(turns)
        print("links passed")
        return Network(links, nodes, turns, self.tot_links, self.tot_nodes, self.tot_turns)

    def __build_nodes(self):
        link_ids = np.arange(self.g.number_of_edges(), dtype=np.uint32)
        values, col, row = csr_prep(np.column_stack((self.node_adjacency[:, 0], link_ids)), self.node_adjacency[:, 1],
                                    (self.tot_nodes, self.tot_links))
        # Note: links are labelled consecutively in the order of their start nodes
        # node 0 has outgoing link(s) [0]
        # node 1 outgoing link(s)  [1,2] and so on
        out_links = UI32CSRMatrix(values, col, row)
        values, col, row = csr_prep(np.column_stack((self.node_adjacency[:, 1], link_ids)),
                                    self.node_adjacency[:, 0], (self.tot_nodes, self.tot_links))
        in_links = UI32CSRMatrix(values, col, row)
        capacity = np.full(self.tot_nodes, node_capacity_default, dtype=np.float32)
        control_type = np.full(self.tot_nodes, node_control_default, dtype=np.int8)
        # add boolean centroid array, alter control type(?) (if necessary)
        number_of_out_links = [len(in_links.get_row(row)) for row in
                               np.arange(self.g.number_of_nodes(), dtype=np.uint32)]
        number_of_in_links = [len(out_links.get_row(row)) for row in
                              np.arange(self.g.number_of_nodes(), dtype=np.uint32)]
        number_of_out_links = np.array(number_of_out_links, dtype=np.uint32)
        number_of_in_links = np.array(number_of_in_links, dtype=np.uint32)
        return Nodes(in_links, out_links, number_of_out_links, number_of_in_links, control_type, capacity)

    def __build_turns(self, nodes: Nodes):
        to_nodes = List()
        from_nodes = List()
        from_links = List()
        to_links = List()
        via_nodes = List()
        turn_counter = 0
        for via_node in np.arange(self.tot_nodes):
            # named here _attribute to indicate all the to nodes/links that are associated with the via_node
            # turns are labelled here topologically by their respective in_link labels, out_links are tiebreakers.

            _to_nodes = nodes.out_links.get_row(via_node)
            _from_nodes = nodes.in_links.get_row(via_node)
            _from_links = nodes.in_links.get_nnz(via_node)
            _to_links = nodes.out_links.get_nnz(via_node)
            for from_node, from_link in zip(_from_nodes, _from_links):
                for to_node, to_link in zip(_to_nodes, _to_links):
                    via_nodes.append(via_node)
                    to_nodes.append(to_node)
                    from_nodes.append(from_node)
                    from_links.append(from_link)
                    to_links.append(to_link)
                    turn_counter += 1
        number_of_turns = turn_counter
        self.tot_turns = number_of_turns
        capacity = np.full(number_of_turns, turn_capacity_default, dtype=np.float32)
        turn_type = np.full(number_of_turns, turn_type_default, dtype=np.int8)
        t0 = np.full(number_of_turns, turn_t0_default, dtype=np.float32)
        return Turns(t0, capacity, np.array(from_nodes, dtype=np.uint32),
                     np.array(via_nodes, dtype=np.uint32),
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
        link_type = np.zeros(self.g.number_of_edges(), dtype=np.int8)  # 0 indicates regular road network link
        # 1 is for sources (connectors leading out of a centroid)
        # -1 for sinks (connectors leading towards a centroid)
        length = np.empty(self.g.number_of_edges(), dtype=np.float32)
        capacity = np.empty(self.g.number_of_edges(), dtype=np.float32)
        v0_prt = np.empty(self.g.number_of_edges(), dtype=np.float32)
        lanes = np.empty(self.g.number_of_edges(), dtype=np.uint8)
        from_node = self.node_adjacency[:, 0]
        to_node = self.node_adjacency[:, 1]
        link_ids = np.arange(self.g.number_of_edges())
        # fix for connectors here, add link types
        for _id, arr in zip(link_ids, self.link_label):
            u, v = arr
            length[_id] = self.g[u][v]['length']
            capacity[_id] = self.g[u][v]['capacity']
            v0_prt[_id] = self.g[u][v]['maxspeed']
            lanes[_id] = self.g[u][v]['lanes']
            if 'connector' in self.g[u][v] and 'centroid' in self.g.nodes[v]:
                link_type[_id] = -1  # sink
            if 'connector' in self.g[u][v] and 'centroid' in self.g.nodes[v]:
                link_type[_id] = 1  # source

        costs = np.empty((self.number_of_time_steps, self.tot_links), dtype=np.float32)
        v_wave = np.full(self.tot_links, v_wave_default, dtype=np.float32)
        number_of_turns = np.uint32(len(turns.to_link))
        fw_index_array = np.column_stack((turns.from_link, turns.to_link))
        bw_index_array = np.column_stack((turns.to_link, turns.from_link))
        val = np.arange(number_of_turns, dtype=np.uint32)
        val, col, row = csr_prep(fw_index_array, val, (self.tot_links, self.tot_links))
        forward = UI32CSRMatrix(val, col, row)
        val, col, row = csr_prep(bw_index_array, val, (self.tot_links, self.tot_links))
        backward = UI32CSRMatrix(val, col, row)

        return Links(length, from_node, to_node, capacity, v_wave, costs, v0_prt, forward, backward,
                     lanes, link_type)

    def __build_demand(self):
        g = self.g
        if 'od_graph' not in g.graph:
            raise ValueError('No od_graph registered on the road graph, see demand.py for required format')
        od_graphs = g.graph['od_graph']
        insertion_time = np.array([od_graph.graph['time'] for od_graph in od_graphs])
        demand_data = [nx.to_scipy_sparse_matrix(c, weight='flow', format='lil') for c in od_graphs]
        # change to csr for consistency ..
        return _build_demand(demand_data, insertion_time, simulation_time=self.time)
    @staticmethod
    def __init_time_obj(time:SimulationTime):
        if time.step_size==parameters.route_choice.step_size:
            route_choice_time=time
            consistent_time =  np.bool_(True)
        else:
            route_choice_time = SimulationTime(time.start, time.end, parameters.route_choice.step_size)
            consistent_time = np.bool_(False)

        @dataclass
        class DTATime:
            network_loading = time
            route_choice = route_choice_time
            consistent :np.bool = consistent_time

        return DTATime()


@dataclass()
class AssignmentResults:
    skims: np.ndarray
    link_costs: np.ndarray
    flows: np.ndarray

# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise staionary characteristics
# example ramp metering event capacity choke, relative and absolute events

# TODO: maybe different link labelling orders that can be set based on the algorithm at hand
# some have an outlink pattern like Algorithm B and others have an inlink outlink pattern like LTM ..
# TODO: how high a priority is visualization.. - one time, interval visualization
# arrays for different things separately by node id
# what does the node data structure look like, turning fraction matrix
# data structure for events a sparse matrix .. time steps( row) * links(columns) value as capacity,
# different ones for each theme

# is the results object a good decision?, we can just pass on network/ demand objects for warm starting
# to enable syntax like DynamicAssignment.Network.Flows
# time incoming outgoing link, belgium netwo
