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
from core.supply import Links, Nodes, Network, Turns
from core.demand import Demand, InternalDynamicDemand
from core.time import SimulationTime
from demand import _check_centroid_connectivity
from settings import parameters
from dataclasses import dataclass
from demand import DynamicDemand
from typing import Callable

v_wave_default = parameters.supply.v_wave_default
turn_capacity_default = parameters.supply.turn_capacity_default
turn_type_default = parameters.supply.turn_type_default
node_capacity_default = parameters.supply.node_capacity_default
turn_t0_default = parameters.supply.turn_t0_default
node_control_default = parameters.supply.node_control_default
network_loading_method = parameters.network_loading.link_model


class Assignment:
    """This class has no value when instantiated on its own,
     it's basically just an interface class that takes all the information from the nx.MultiDiGraph and the
     DynamicDemand and translates it into internal representations that can be understood by numba
     """

    def __init__(self, g: nx.DiGraph, dynamic_demand: DynamicDemand, simulation_time: SimulationTime):
        """

        Parameters
        ----------
        g : nx.MultiDiGraph
        dynamic_demand : DynamicDemand
        """
        # the data structures starting with _ refer to internal compiled structures, if you want to change them
        # you have to be familiar with numba
        _check_centroid_connectivity(g)
        self.g = g
        self.dynamic_demand = dynamic_demand
        self.time = self.__init_time_obj(simulation_time)
        # get adjacency from nx, and
        self.number_of_time_steps = np.uint32(10)
        # self.demand = self.build_demand()
        self.nb_network = self.__build_network()
        print('network build')
        self.nb_dynamic_demand: InternalDynamicDemand = self._build_internal_dynamic_demand(dynamic_demand,
                                                                                            simulation_time)
        print('demand simulation build')

    def run(self, method: Callable):
        from core.assignment_methods.methods import valid_methods
        # TODO: generic way for adding keyword args
        results: AssignmentResults = method(self.nb_network, self.nb_dynamic_demand, self.time.route_choice, self.time.network_loading)
        return results

    @staticmethod
    def get_methods():
        from core.assignment_methods.methods import valid_methods
        return valid_methods

    def __build_network(self):
        sorted_edges = sorted(self.g.edges(data=True), key=lambda t: t[2]['link_id'])
        node_ids = np.array([node_id for (_, node_id) in self.g.nodes.data('node_id')], dtype=np.uint32)
        # for the future: remove this requirement of pre sorting of nodes.
        if not np.all(node_ids[1:] >= node_ids[:-1]):
            raise ValueError('the node_ids in the graph are assumed to be monotonously increasing and have to be '
                             'added accordingly')
        tot_nodes = np.uint32(self.g.number_of_nodes())
        tot_links = np.uint32(self.g.number_of_edges())
        from_nodes = np.array([d['from_node_id'] for (_, _, d) in sorted_edges], dtype=np.uint32)
        to_nodes = np.array([d['to_node_id'] for _, _, d in sorted_edges], dtype=np.uint32)
        link_ids = np.arange(tot_links, dtype=np.uint32)

        nodes = self.__build_nodes(tot_nodes, tot_links, from_nodes, to_nodes, link_ids)
        print("nodes passed")
        turns = self.__build_turns(tot_nodes, nodes)
        print("turns passed")

        link_capacity = np.array([d['capacity'] for (_, _, d) in sorted_edges], dtype=np.uint32)
        free_speed = np.array([d['free_speed'] for (_, _, d) in sorted_edges], dtype=np.float32)
        lanes = np.array([d['lanes'] for (_, _, d) in sorted_edges], dtype=np.uint8)
        length = np.array([d['length'] for (_, _, d) in sorted_edges], dtype=np.float32)
        link_type = np.array([np.int8(d.get('link_type', 0)) for (_, _, d) in sorted_edges], dtype=np.int8)
        tot_source_connectors = np.argwhere(link_type==1).size
        tot_sink_connectors =  np.argwhere(link_type==-1).size
        # 1 is for sources (connectors leading out of a centroid)
        # -1 for sinks (connectors leading towards a centroid)
        tot_time_steps = self.time.network_loading.tot_time_steps
        links = self.__build_links(turns, tot_time_steps, tot_links, from_nodes, to_nodes, link_capacity, free_speed,
                                   lanes, length, link_type)
        print("links passed")
        return Network(links, nodes, turns, self.g.number_of_edges(), self.g.number_of_nodes(), turns.capacity.size, tot_source_connectors, tot_sink_connectors)

    @staticmethod
    def __build_nodes(tot_nodes, tot_links, from_nodes, to_nodes, link_ids):
        values, col, row = csr_prep(np.column_stack((from_nodes, link_ids)), to_nodes,
                                    (tot_nodes, tot_links))
        # Note: links are labelled consecutively in the order of their start nodes
        # node 0 has outgoing link(s) [0]
        # node 1 outgoing link(s)  [1,2] and so on
        out_links = UI32CSRMatrix(values, col, row)
        values, col, row = csr_prep(np.column_stack((to_nodes, link_ids)),
                                    from_nodes, (tot_nodes, tot_links))
        in_links = UI32CSRMatrix(values, col, row)
        capacity = np.full(tot_nodes, node_capacity_default, dtype=np.float32)
        control_type = np.full(tot_nodes, node_control_default, dtype=np.int8)
        # add boolean centroid array, alter control type(?) (if necessary)
        number_of_out_links = [len(in_links.get_row(row)) for row in
                               np.arange(tot_nodes, dtype=np.uint32)]
        number_of_in_links = [len(out_links.get_row(row)) for row in
                              np.arange(tot_nodes, dtype=np.uint32)]
        number_of_out_links = np.array(number_of_out_links, dtype=np.uint32)
        number_of_in_links = np.array(number_of_in_links, dtype=np.uint32)
        return Nodes(in_links, out_links, number_of_out_links, number_of_in_links, control_type, capacity)

    @staticmethod
    def __build_turns(tot_nodes, nodes: Nodes):
        to_nodes = List()
        from_nodes = List()
        from_links = List()
        to_links = List()
        via_nodes = List()
        turn_counter = 0
        for via_node in np.arange(tot_nodes):
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
        capacity = np.full(number_of_turns, turn_capacity_default, dtype=np.float32)
        turn_type = np.full(number_of_turns, turn_type_default, dtype=np.int8)
        t0 = np.full(number_of_turns, turn_t0_default, dtype=np.float32)
        return Turns(t0, capacity, np.array(from_nodes, dtype=np.uint32),
                     np.array(via_nodes, dtype=np.uint32),
                     np.array(to_nodes, dtype=np.uint32), np.array(from_links, dtype=np.uint32),
                     np.array(to_links, dtype=np.uint32), turn_type)

    @staticmethod
    def __build_links(turns, tot_time_steps, tot_links, from_nodes, to_nodes, capacity, free_speed, lanes, length,
                      link_type):
        """
        initiates all the different numpy arrays for the links object from nx.DiGraph,
        requires the networkx graph to be set up as specified in the network_data
        Returns
        links : Links
        -------

        """
        costs = np.empty((tot_time_steps, tot_links), dtype=np.float32)
        v_wave = np.full(tot_links, v_wave_default, dtype=np.float32)
        number_of_turns = np.uint32(len(turns.to_link))
        fw_index_array = np.column_stack((turns.from_link, turns.to_link))
        bw_index_array = np.column_stack((turns.to_link, turns.from_link))
        val = np.arange(number_of_turns, dtype=np.uint32)
        val, col, row = csr_prep(fw_index_array, val, (tot_links, tot_links))
        forward = UI32CSRMatrix(val, col, row)
        val, col, row = csr_prep(bw_index_array, val, (tot_links, tot_links))
        backward = UI32CSRMatrix(val, col, row)

        return Links(length, from_nodes, to_nodes, capacity, v_wave, costs, free_speed, forward, backward,
                     lanes, link_type)

    @staticmethod
    def __init_time_obj(time: SimulationTime):
        if time.step_size == parameters.route_choice.step_size:
            route_choice_time = time
            consistent_time = np.bool_(True)
        else:
            route_choice_time = SimulationTime(time.start, time.end, parameters.route_choice.step_size)
            consistent_time = np.bool_(False)

        @dataclass
        class DTATime:
            network_loading = time
            route_choice = route_choice_time
            consistent: np.bool = consistent_time

        return DTATime()

    @staticmethod
    def _build_internal_dynamic_demand(dynamic_demand: DynamicDemand, simulation_time: SimulationTime):
        """

        Parameters
        ----------
        dynamic_demand: DynamicDemand as defined in demand.py

        Returns
        -------

        """
        # finding closest time step for defined demand insertion times
        # each time is translated to an index and element of [0,1, ..., tot_time_steps]
        insertion_times = np.array(
            [np.argmin(np.abs(np.arange(simulation_time.tot_time_steps) * simulation_time.step_size - time)) for time in
             dynamic_demand.insertion_times], dtype=np.uint32)
        demand_data = [dynamic_demand.get_sparse_repr(t) for t in dynamic_demand.insertion_times]

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
        cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
        tot_centroids = np.uint32(
            max([trip_graph.number_of_nodes() for _, trip_graph in dynamic_demand.trip_graphs.items()]))
        for internal_time, lil_demand, row, col in zip(loading_time_steps, demand_data, rows, cols):
            vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
            index_array_to_d = np.column_stack((row, col))
            index_array_to_o = np.column_stack((col, row))
            to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (tot_centroids, tot_centroids)))
            to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (tot_centroids, tot_centroids)))
            static_demands.append(Demand(to_origins, to_destinations,
                                         to_origins.get_nnz_rows(), to_destinations.get_nnz_rows(),
                                         np.uint32(internal_time)))
        return InternalDynamicDemand(static_demands, simulation_time.tot_time_steps, tot_centroids)


@dataclass()
class AssignmentResults:
    skims: np.ndarray
    link_costs: np.ndarray
    flows: np.ndarray

# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise stationary characteristics
# example ramp metering event capacity choke, relative and absolute events
