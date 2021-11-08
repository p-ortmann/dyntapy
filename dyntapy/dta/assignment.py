#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
#
import numpy as np
import networkx as nx
from numba.typed import List

import dyntapy.assignment_context
from dyntapy.datastructures.csr import csr_prep, UI32CSRMatrix, F32CSRMatrix, csr_sort
from dyntapy.dta.core.supply import Links, Nodes, Network, Turns
from dyntapy.dta.core.demand import InternalDynamicDemand
from dyntapy.sta.demand import Demand
from dyntapy.dta.core.time import SimulationTime
from dyntapy.demand import _check_centroid_connectivity
from dyntapy.dta.core.assignment_methods.i_ltm_aon import i_ltm_aon
from dyntapy.dta.core.assignment_methods.aon import aon
from dyntapy.dta.core.assignment_methods.incremental_assignment import incremental
from dyntapy.settings import dynamic_parameters
from dataclasses import dataclass
from dyntapy.demand import DynamicDemand
from dyntapy.utilities import log
from warnings import warn

v_wave_default = dynamic_parameters.supply.v_wave_default
turn_capacity_default = dynamic_parameters.supply.turn_capacity_default
turn_type_default = dynamic_parameters.supply.turn_type_default
node_capacity_default = dynamic_parameters.supply.node_capacity_default
turn_t0_default = dynamic_parameters.supply.turn_t0_default
node_control_default = dynamic_parameters.supply.node_control_default
network_loading_method = dynamic_parameters.network_loading.link_model


class Assignment:
    """This class stores all the information needed for the assignment itself.
     It takes all the information from the nx.MultiDiGraph and the
     DynamicDemand and translates it into internal representations that can be understood by numba.
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
        self.time = simulation_time
        # get adjacency from nx, and
        # self.demand = self.build_demand()
        self.internal_network = build_network(g)
        log('network build')

        self.internal_dynamic_demand: InternalDynamicDemand = self._build_internal_dynamic_demand(dynamic_demand,
                                                                                                  simulation_time,
                                                                                                  self.internal_network)
        log('demand simulation build')

    def run(self, method: str = 'i_ltm_aon'):
        dyntapy.assignment_context.running_assignment = self  # making the current assignment available as global var
        methods = {'i_ltm_aon': i_ltm_aon,
                   'incremental_assignment': incremental,
                   'aon': aon}
        if method in methods:
            flows, costs = methods[method](self.internal_network, self.internal_dynamic_demand, self.time)
        else:
            raise NotImplementedError(f'{method=} is not defined ')
        return flows, costs

    @staticmethod
    def __init_time_obj(time: SimulationTime):
        route_choice_time = time  # for now network loading and route choice time are always equal

        @dataclass
        class DTATime:
            network_loading = time
            route_choice = route_choice_time

        return DTATime()

    @staticmethod
    def _build_internal_dynamic_demand(dynamic_demand: DynamicDemand, simulation_time: SimulationTime,
                                       network: Network):
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

        static_demands = List()
        rows = [np.asarray(lil_demand.nonzero()[0], dtype=np.uint32) for lil_demand in demand_data]
        cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
        tot_centroids = np.uint32(
            max([trip_graph.number_of_nodes() for trip_graph in dynamic_demand.od_graphs]))
        for internal_time, lil_demand, row, col in zip(insertion_times, demand_data, rows, cols):
            vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
            index_array_to_d = np.column_stack((row, col))
            index_array_to_o = np.column_stack((col, row))
            to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (tot_centroids, tot_centroids)))
            to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (tot_centroids, tot_centroids)))
            static_demands.append(Demand(to_origins, to_destinations,
                                         to_destinations.get_nnz_rows(), to_origins.get_nnz_rows(),
                                         np.uint32(internal_time)))
        return InternalDynamicDemand(static_demands, simulation_time.tot_time_steps, tot_centroids,
                                     network.nodes.in_links)


def build_network(g):
    edge_data = [(_, _, data) for _, _, data in g.edges.data()]
    sorted_edges = sorted(edge_data, key=lambda t: t[2]['link_id'])
    sorted_nodes = sorted(g.nodes(data=True), key=lambda t: t[1]['node_id'])
    node_ids = np.array([data['node_id'] for (_, data) in sorted_nodes], dtype=np.uint32)
    # for the future: remove this requirement of pre sorting of nodes.
    if not np.all(node_ids[1:] == node_ids[:-1] + 1):
        raise ValueError('the node_ids in the graph are assumed to be monotonously increasing and have to be '
                         'added accordingly')
    tot_nodes = np.uint32(g.number_of_nodes())
    tot_links = np.uint32(g.number_of_edges())
    from_nodes = np.array([d['from_node_id'] for (_, _, d) in sorted_edges], dtype=np.uint32)
    to_nodes = np.array([d['to_node_id'] for _, _, d in sorted_edges], dtype=np.uint32)
    link_ids = np.array([d['link_id'] for _, _, d in sorted_edges], dtype=np.uint32)
    if not np.all(link_ids[1:] == link_ids[:-1] + 1):
        raise ValueError('the node_ids in the graph are assumed to be monotonously increasing and have to be '
                         'added accordingly')

    nodes = build_nodes(tot_nodes, tot_links, from_nodes, to_nodes, link_ids)
    log("nodes passed")
    link_type = np.array([np.int8(d.get('link_type', 0)) for (_, _, d) in sorted_edges], dtype=np.int8)
    turns = build_turns(tot_nodes, nodes, link_type)
    log("turns passed")

    link_capacity = np.array([d['capacity'] for (_, _, d) in sorted_edges], dtype=np.float32)
    free_speed = np.array([d['free_speed'] for (_, _, d) in sorted_edges], dtype=np.float32)
    lanes = np.array([d['lanes'] for (_, _, d) in sorted_edges], dtype=np.uint8)
    length = np.array([d['length'] for (_, _, d) in sorted_edges], dtype=np.float32)
    max_length = np.max(length)
    if np.max(length) > 100:
        warn(f'Network contains very long links, up to {max_length} km. Implementation has not been verified for'
             f'this type of network. calculations may yield unexpected results.')

    tot_connectors = np.argwhere(link_type == 1).size + np.argwhere(link_type == -1).size
    # 1 is for sources (connectors leading out of a centroid)
    # -1 for sinks (connectors leading towards a centroid)
    links = build_links(turns, tot_links, from_nodes, to_nodes, link_capacity, free_speed,
                        lanes, length, link_type)
    log("links passed")

    return Network(links, nodes, turns, g.number_of_edges(),
                   g.number_of_nodes(), turns.capacity.size,
                   tot_connectors)


def build_nodes(tot_nodes, tot_links, from_nodes, to_nodes, link_ids):
    values, col, row = csr_prep(np.column_stack((from_nodes, link_ids)), to_nodes,
                                (tot_nodes, tot_links))
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
    return Nodes(out_links, in_links, number_of_out_links, number_of_in_links, control_type, capacity)


def build_turns(tot_nodes, nodes: Nodes, link_types):
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
                if not (link_types[from_link] == -1 and link_types[to_link] == 1):
                    # u turns are allowed
                    # excluding turns that go from sink to source connectors and vice versa
                    via_nodes.append(via_node)
                    to_nodes.append(to_node)
                    from_nodes.append(from_node)
                    from_links.append(from_link)
                    to_links.append(to_link)
                    turn_counter += 1
    fw_index_array = np.column_stack((from_links, to_links))
    turn_order = np.arange(turn_counter)
    res, turn_order = csr_sort(fw_index_array, turn_order, turn_counter)
    via_nodes = np.array(via_nodes, dtype=np.uint32)
    to_nodes = np.array(to_nodes, dtype=np.uint32)
    from_links = np.array(from_links, dtype=np.uint32)
    to_links = np.array(to_links, dtype=np.uint32)

    def sort(arr, order):
        tmp = np.empty_like(arr)
        for i, j in enumerate(order):
            tmp[i] = arr[j]
        return tmp

    via_nodes = sort(via_nodes, turn_order)
    from_nodes = sort(from_nodes, turn_order)
    to_nodes = sort(to_nodes, turn_order)
    from_links = sort(from_links, turn_order)
    to_links = sort(to_links, turn_order)
    number_of_turns = turn_counter
    t0 = np.full(number_of_turns, turn_t0_default, dtype=np.float32)
    capacity = np.full(number_of_turns, turn_capacity_default, dtype=np.float32)
    turn_type = np.full(number_of_turns, turn_type_default, dtype=np.int8)
    return Turns(t0, capacity, np.array(from_nodes, dtype=np.uint32),
                 np.array(via_nodes, dtype=np.uint32),
                 np.array(to_nodes, dtype=np.uint32), np.array(from_links, dtype=np.uint32),
                 np.array(to_links, dtype=np.uint32), turn_type)


def build_links(turns, tot_links, from_nodes, to_nodes, capacity, free_speed, lanes, length,
                link_type):
    """
    initiates all the different numpy arrays for the links object from nx.DiGraph,
    requires the networkx graph to be set up as specified in the network_data
    Returns
    links : Links
    -------

    """
    length[length < 0.05] = 0.05
    v_wave = np.full(tot_links, v_wave_default, dtype=np.float32)
    tot_turns = np.uint32(len(turns.to_link))
    fw_index_array = np.column_stack((turns.from_link, np.arange(tot_turns, dtype=np.uint32)))
    bw_index_array = np.column_stack((turns.to_link, np.arange(tot_turns, dtype=np.uint32)))
    val = turns.to_link
    val, col, row = csr_prep(fw_index_array, val, (tot_links, tot_turns), unsorted=False)
    out_turns = UI32CSRMatrix(val, col, row)
    val = np.copy(turns.from_link)
    val, col, row = csr_prep(bw_index_array, val, (tot_links, tot_turns))
    in_turns = UI32CSRMatrix(val, col, row)

    return Links(length, from_nodes, to_nodes, capacity, v_wave, free_speed, out_turns, in_turns,
                 lanes, link_type)


@dataclass()
class AssignmentResults:
    skims: np.ndarray
    link_costs: np.ndarray
    flows: np.ndarray

# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise stationary characteristics
# example ramp metering event capacity choke, relative and absolute events
