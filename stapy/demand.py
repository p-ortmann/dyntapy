#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#

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
