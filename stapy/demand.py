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
import numpy as np


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
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=str(int_dtype))
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
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=str(int_dtype))
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
