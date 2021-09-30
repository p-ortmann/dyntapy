#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import networkx as nx
from numba.experimental import jitclass
from scipy.sparse import lil_matrix
from numba.typed import List, Dict
import numpy as np
from collections import OrderedDict
from numba import uint32
from dyntapy.datastructures.csr import f32csr_type, F32CSRMatrix, csr_prep


def build_demand_structs(od_matrix):
    demand_dict = Dict()
    od_flow_vector = []
    origins = set(od_matrix.nonzero()[0].astype(np.uint32))
    for i in origins:
        my_list = List()
        destinations = od_matrix.getrow(i).tocoo().col.astype(np.uint32)
        demands = od_matrix.getrow(i).tocoo().data.astype(np.uint32)
        # discarding intrazonal traffic ..
        origin_index = np.where(destinations == i)
        destinations = np.delete(destinations, origin_index)
        demands = np.delete(demands, origin_index)
        assert len(destinations) == len(demands)
        for demand in demands:
            od_flow_vector.append(np.float32(demand))
        my_list.append(destinations)
        my_list.append(demands)
        demand_dict[i] = my_list
    od_flow_vector = np.array(od_flow_vector)
    return demand_dict, od_flow_vector


def generate_od(number_of_nodes, origins_to_nodes_ratio, origins_to_destinations_connection_ratio=0.15, seed=0):
    """

    Parameters
    ----------
    number_of_nodes : number of nodes (potential origins)
    origins_to_nodes_ratio : float, indicates what fraction of nodes are assumed to be origins
    seed : seed for numpy random
    origins_to_destinations_connection_ratio :

    Returns
    -------
    od_matrix

    """
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    od_connections = int(origins_to_destinations_connection_ratio)
    number_of_origins = int(origins_to_nodes_ratio * number_of_nodes)
    origins = np.random.choice(np.arange(number_of_nodes), size=number_of_origins, replace=False)
    destinations = np.random.choice(np.arange(number_of_nodes), size=number_of_origins, replace=False)
    for origin in origins:
        # randomly sample how many and which destinations this origin is connected to
        number_of_destinations = int(np.random.gumbel(loc=origins_to_destinations_connection_ratio,
                                                      scale=origins_to_destinations_connection_ratio / 2) * len(
            destinations))
        if number_of_destinations < 0: continue
        if number_of_destinations > len(destinations): number_of_destinations = len(destinations)
        destinations_by_origin = np.random.choice(destinations, size=number_of_destinations, replace=False)
        rand_od.rows[origin] = list(destinations_by_origin)
        rand_od.data[origin] = [int(np.random.random() * 2000) for _ in destinations_by_origin]
    return rand_od


def generate_od_fixed(number_of_nodes, number_of_od_values, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes)
    vals = np.random.choice(arr, size=number_of_od_values, replace=False)
    ids = [np.where(arr.reshape((number_of_nodes, number_of_nodes)) == val) for val in vals]
    for i, j in ids:
        i, j = int(i), int(j)
        if isinstance(rand_od.rows[i], list):
            rand_od.rows[i].append(j)
            rand_od.data[i].append(int(np.random.random() * 2000))
        else:
            rand_od.rows[i] = list(j)
            rand_od.data[i] = list((int(np.random.random() * 2000)))
    return rand_od


def generate_random_bush(number_of_nodes, number_of_branches, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes)
    origin = np.random.randint(0, number_of_nodes)
    destinations = np.random.choice(np.arange(0, number_of_nodes), number_of_branches, replace=False)
    for destination in destinations:
        i, j = int(origin), int(destination)
        if isinstance(rand_od.rows[i], list):
            rand_od.rows[i].append(j)
            rand_od.data[i].append(int(np.random.random() * 2000))
        else:
            rand_od.rows[i] = list(j)
            rand_od.data[i] = list((int(np.random.random() * 2000)))
    return rand_od


spec_demand = [('to_destinations', f32csr_type),
               ('to_origins', f32csr_type),
               ('origins', uint32[:]),
               ('destinations', uint32[:]),
               ('time_step', uint32)]
spec_demand = OrderedDict(spec_demand)


def build_static_demand(od_graph: nx.DiGraph):
    lil_demand = nx.to_scipy_sparse_matrix(od_graph, weight='flow', format='lil')
    tot_centroids = od_graph.number_of_nodes()
    row = np.asarray(lil_demand.nonzero()[0])
    col = np.asarray(lil_demand.nonzero()[1])
    vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
    index_array_to_d = np.column_stack((row, col))
    index_array_to_o = np.column_stack((col, row))
    to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (tot_centroids, tot_centroids)))
    to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (tot_centroids, tot_centroids)))
    return Demand(to_origins, to_destinations,
                  to_destinations.get_nnz_rows(), to_origins.get_nnz_rows(),
                  np.uint32(0))


@jitclass(spec_demand)
class Demand(object):
    def __init__(self, to_origins: F32CSRMatrix, to_destinations: F32CSRMatrix, origins, destinations,
                 time_step):
        self.to_destinations = to_destinations  # csr matrix origins x destinations
        self.to_origins = to_origins  # csr destinations x origins
        self.origins = origins  # array of active origin id's
        self.destinations = destinations  # array of active destination id's
        self.time_step = time_step  # time at which this demand is added to the network


def get_demand_fraction(demand: Demand, fraction=np.float):
    # returns new demand object with fraction x cell for all cells.
    assert fraction > 0.0
    values = np.copy(demand.to_destinations.values)
    values = values * fraction
    to_destinations = F32CSRMatrix(values, demand.to_destinations.col_index, demand.to_destinations.row_index)
    values = np.copy(demand.to_origins.values)
    values = values * fraction
    to_origins = F32CSRMatrix(values, demand.to_origins.col_index, demand.to_origins.row_index)
    return Demand(to_origins, to_destinations, demand.origins, demand.destinations, demand.time_step)
