#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
import networkx as nx
from numba.typed import List
from datastructures.csr import csr_prep, UI32CSRMatrix, F32CSRMatrix
from core.assignment_cls import Links, Nodes, Network, Turns, InternalDynamicDemand, SimulationTime, StaticDemand
from demand import _check_centroid_connectivity
from settings import parameters
from dataclasses import dataclass
from demand import DynamicDemand
from typing import Callable
from core.assignment_methods.i_ltm_aon import i_ltm_aon


@dataclass
class valid_methods:
    i_ltm_aon: Callable = i_ltm_aon



v_wave_default = parameters.supply.v_wave_default
turn_capacity_default = parameters.supply.turn_capacity_default
turn_type_default = parameters.supply.turn_type_default
node_capacity_default = parameters.supply.node_capacity_default
turn_t0_default = parameters.supply.turn_t0_default
node_control_default = parameters.supply.node_control_default
network_loading_method = parameters.network_loading.method


class Assignment:
    """This class has no value when instantiated on its own,
     it's basically just an interface class that takes all the information from the nx.MultiDiGraph and the
     DynamicDemand and translates it into internal representations that can be understood by numba
     """

    def __init__(self, g: nx.DiGraph, dynamic_demand: DynamicDemand,
                 method=valid_methods.i_ltm_aon):
        """

        Parameters
        ----------
        g : nx.DiGraph
        dynamic_demand : DynamicDemand
        """
        # the data structures starting with _ refer to internal compiled structures, if you want to change them
        # you have to be familiar with numba
        self.method = method
        _check_centroid_connectivity(g)
        self.g = g
        self.dynamic_demand= dynamic_demand
        self.time = self.__init_time_obj(dynamic_demand.simulation_time)
        #get adjacency from nx, and
        self.number_of_time_steps = np.uint32(10)
        # self.demand = self.build_demand()
        self._network = self.__build_network()
        print('network build')
        self._dynamic_demand: InternalDynamicDemand = self._build_internal_dynamic_demand(dynamic_demand)
        print('demand simulation build')
        self.results=self.run_assignment()

        print('DNL data structures build')
        # replacing network object with specialized version for network loading method
        print('network build, lets see')
        # dict of dict with outer dict keyed by edges (u,v) and inner dict as data
        # to be dumped into the nx.DiGraph as key value pairs
        # self.set_od_matrix(od_matrix)

    def run_assignment(self):
        #TODO: generic way for adding keyword args
        results: AssignmentResults=self.method(self)
        return results

    def __build_network(self):
        nodes = self.__build_nodes()
        print("nodes passed")
        turns = self.__build_turns(nodes)
        print("turns passed")
        links = self.__build_links(turns)
        print("links passed")
        return Network(links, nodes, turns, self.g.number_of_edges(), self.g.number_of_nodes(), self.tot_turns)

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
            consistent:np.bool = consistent_time

        return DTATime()
    @staticmethod
    def _build_internal_dynamic_demand(dynamic_demand: DynamicDemand):
        """

        Parameters
        ----------
        dynamic_demand: DynamicDemand as defined above

        Returns
        -------

        """
        insertion_times = dynamic_demand.insertion_times
        demand_data = [dynamic_demand.get_sparse_repr(t) for t in dynamic_demand.insertion_times]
        simulation_time = dynamic_demand.simulation_time

        if not np.all(insertion_times[1:] - insertion_times[:-1] > simulation_time.step_size):
            raise ValueError(
                'insertion times are assumed to be monotonously increasing. The minimum difference between '
                'two '
                'insertions is the internal simulation time step')
        if max(insertion_times > 24):
            raise ValueError('internally time is restricted to 24 hours')
        time = np.arange(simulation_time.start, simulation_time.end, simulation_time.step_size)
        loading_time_steps = [(np.abs(insertion_time - time)).argmin() for insertion_time in insertion_times]
        static_demands = List()
        rows = [np.asarray(lil_demand.nonzero()[0], dtype=np.uint32) for lil_demand in demand_data]
        row_sizes = np.array([lil_demand.nonzero()[0].size for lil_demand in demand_data], dtype=np.uint32)
        cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
        col_sizes = np.array([lil_demand.nonzero()[1].size for lil_demand in demand_data], dtype=np.uint32)
        all_destinations, cols = np.unique(np.concatenate(cols), return_inverse=True)
        all_origins, rows = np.unique(np.concatenate(rows), return_inverse=True)
        cols = np.array_split(cols, np.cumsum(col_sizes))
        rows = np.array_split(rows, np.cumsum(row_sizes))
        tot_destinations = max(all_destinations)
        tot_origins = max(all_origins)
        for internal_time, lil_demand, row, col in zip(loading_time_steps, demand_data, rows, cols):
            vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
            index_array_to_d = np.column_stack((row, col))
            index_array_to_o = np.column_stack((col, row))
            to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (tot_origins, tot_destinations)))
            to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (tot_origins, tot_destinations)))
            origin_node_ids = np.array([all_origins[i] for i in to_destinations.get_nnz_rows()], dtype=np.uint32)
            destination_node_ids = np.array([all_destinations[i] for i in to_origins.get_nnz_rows()], dtype=np.uint32)
            static_demands.append(StaticDemand(to_origins, to_destinations,
                                               to_origins.get_nnz_rows(), to_destinations.get_nnz_rows(),
                                               origin_node_ids,
                                               destination_node_ids, internal_time))
        return InternalDynamicDemand(static_demands, simulation_time.tot_time_steps, all_origins, all_destinations)


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
