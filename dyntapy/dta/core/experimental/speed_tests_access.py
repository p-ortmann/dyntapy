#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import networkx as nx
import pandas as pd
import os
import numpy as np
from itertools import count
import numba as nb
import timeit
from dyntapy.dta.core.experimental.dict_forward_star import make_forward_stars
from dyntapy.datastructures.csr import csr_prep, UI32CSRMatrix

# this file compares access times forward stars among three different implementation methods:
# (i) via numba's typed dict, (2) as a custom implementation of CSR (3) as linked list as shown by graphhopper, see
# https://github.com/graphhopper/graphhopper/blob/master/docs/core/technical.md

# we use here a rather large network file for belgium that contains over 600k nodes
df = pd.read_csv(os.getcwd() + os.path.sep + 'data' + os.path.sep + 'belgium.csv')
Graphtype = nx.DiGraph()
g = nx.from_pandas_edgelist(df, edge_attr=['length', 'highway', 'osmid', 'maxspeed'], source='to', target='from',
                            create_using=Graphtype)
edge_array = np.empty((g.number_of_edges(), 2), dtype=str(nb.int64))
counter = count()
for node_id, u in enumerate(g.nodes):
    g.nodes[u]['_id'] = node_id
    for v in g.succ[u]:
        link_id = next(counter)
        g[u][v]['_id'] = link_id
for u, v, link_id in g.edges.data('_id'):
    _u, _v = g.nodes[u]['_id'], g.nodes[v]['_id']
    edge_array[link_id] = _u, _v

dict_forward_star = make_forward_stars(edge_array, g.number_of_nodes())
my_csr_matrix=UI32CSRMatrix(**csr_prep(edge_array, g.number_of_nodes(), (counter,counter)))
my_csr_matrix.get_nnz(10) #gives forward star nodes for node 10
my_csr_matrix.get_row(10) # gives forward star links for node 10
order=np.arange(g.number_of_nodes()).astype(dtype=str(nb.int64))
np.random.shuffle(order)
@nb.njit
def test_csr_access(order, my_csr_matrix):
    for i in order:
        my_csr_matrix.get_nnz(i)
@nb.njit
def test_dict_access(order, my_dict_forward_star):
    for i in order:
        my_dict_forward_star[i]
#force compilation before timing
test_dict_access(order,dict_forward_star)
test_csr_access(order, my_csr_matrix)

print(timeit.timeit(stmt='test_csr_access(order, my_csr_matrix)', setup='from __main__ import my_csr_matrix, order, test_csr_access',
                    number=1000))
print(timeit.timeit(stmt='test_dict_access(order, dict_forward_star)',
                     setup='from __main__ import order, dict_forward_star, test_dict_access', number=1000))
print('done.')