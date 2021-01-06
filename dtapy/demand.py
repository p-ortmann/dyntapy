#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from scipy.sparse import lil_matrix
from stapy.setup import float_dtype, int_dtype
from numba.typed import List, Dict
from collections import namedtuple
import numpy as np
from dtapy.core.jitclasses import SimulationTime, StaticDemand, DynamicDemand
from datastructures.csr import csr_prep, F32CSRMatrix


def build_demand_structs(od_matrix):
    demand_dict = Dict()
    od_flow_vector = []
    origins = set(od_matrix.nonzero()[0].astype(str(int_dtype)))
    for i in origins:
        my_list = List()
        destinations = od_matrix.getrow(i).tocoo().col.astype(str(int_dtype))
        demands = od_matrix.getrow(i).tocoo().data.astype(str(int_dtype))
        # discarding intrazonal traffic ..
        origin_index = np.where(destinations == i)
        destinations = np.delete(destinations, origin_index)
        demands = np.delete(demands, origin_index)
        assert len(destinations) == len(demands)
        for demand in demands:
            od_flow_vector.append(float_dtype(demand))
        my_list.append(destinations)
        my_list.append(demands)
        demand_dict[i] = my_list
    od_flow_vector = np.array(od_flow_vector)
    return demand_dict, od_flow_vector


def generate_od_fixed(number_of_nodes, number_of_od_values, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes, dtype=np.uint32)
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
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=str(int_dtype))
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


# TODO: add a generic way of handling demand from and to geographical locations, willem update method


def build_demand(demand_data, insertion_times, simulation_time: SimulationTime, number_of_nodes):
    """
    
    Parameters
    ----------
    simulation_time : time object, see class def
    demand_data : List <scipy.lil_matrix> # node x node demand, each element is added demand for a particular moment in time
    insertion_times : Array, times at which the demand is loaded

    Returns
    -------

    """
    # internally we maintain different labels for origins and destinations starting at zero,
    # the corresponding node ids are stored in the static demand object
    # this makes it easier to work with destination based flows ( keeping the label space small)

    if not np.all(insertion_times[1:] - insertion_times[:-1] > simulation_time.step_size):
        raise ValueError('insertion times are assumed to be monotonously increasing. The minimum difference between '
                         'two '
                         'insertions is the internal simulation time step')
    times = np.arange(simulation_time.start, simulation_time.end, simulation_time.step_size)
    loading_time_steps = [(np.abs(insertion_time - times)).argmin() for insertion_time in insertion_times]
    static_demands = List()
    rows = [np.asarray(lil_demand.nonzero()[0], dtype=np.uint32) for lil_demand in demand_data]
    row_sizes = np.array([lil_demand.nonzero()[0].size for lil_demand in demand_data], dtype=np.uint32)
    cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
    col_sizes = np.array([lil_demand.nonzero()[1].size for lil_demand in demand_data], dtype=np.uint32)
    all_destinations, cols = np.unique(np.concatenate(cols), return_inverse=True)
    all_origins, rows = np.unique(np.concatenate(rows), return_inverse=True)
    cols = np.array_split(cols, np.cumsum(col_sizes))
    rows = np.array_split(rows, np.cumsum(row_sizes))
    tot_destinations = all_destinations.size
    tot_origins = all_origins.size
    row_counter = 0
    col_counter = 0

    for internal_time, lil_demand, row, col in zip(loading_time_steps, demand_data, rows, cols):
        vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
        index_array_to_d = np.column_stack((row, col))
        index_array_to_o = np.column_stack((col, row))
        to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (number_of_nodes, number_of_nodes)))
        to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (number_of_nodes, number_of_nodes)))
        origin_node_ids = np.array([all_origins[i] for i in to_destinations.get_nnz_rows()], dtype=np.uint32)
        destination_node_ids = np.array([all_destinations[i] for i in to_origins.get_nnz_rows()], dtype=np.uint32)
        static_demands.append(StaticDemand(to_origins, to_destinations,
                                           to_origins.get_nnz_rows(), to_destinations.get_nnz_rows(), origin_node_ids,
                                           destination_node_ids, internal_time))

    return DynamicDemand(static_demands, simulation_time.tot_time_steps, all_origins, all_destinations)
