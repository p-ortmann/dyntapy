#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numba as nb
import numpy as np
from collections import OrderedDict

def csr_sorted(index_array, values):
    pass

@csr_sorted
@nb.njit
def construct_sparse_link_matrix(index_array, values, number_of_nodes):
    # edge_array with the link ids as index, 0th column is the from-node and first column the to-node
    # sorted by from node with ties settled by to node.
    # example:
    # array([[0, 1],
    #        [1, 480640],
    #        [2, 3],
    #        [2, 356104],
    #        [2, 356106],
    #        [3, 4],
    #        [3, 5],
    #        [4, 356112],
    #        [5, 355852],
    #        [5, 356094]], dtype=int64)
    #values=nb.typed.List()
    col, row = nb.typed.List(), nb.typed.List(), nb.typed.List()
    row.append(0)
    row_counter = 0
    link_counter = 0
    link_id = 0
    for i in np.arange(number_of_nodes+1, dtype=np.int64):
        link_counter = 0
        for edge in index_array[link_id:]:
            if i == edge[0]:
                #values.append(link_id+link_counter)
                col.append(index_array[link_id + link_counter][1])
                row_counter += 1
                link_counter += 1
            else:
                # next row
                row.append(row_counter)
                link_id += link_counter
                break

    return CSRMatrix(np.asarray(values, dtype=np.int64), np.asarray(col, dtype=np.int64),
                     np.asarray(row, dtype=np.int64))

    # challenge if you have this ordered all link id's disappear


spec_csr_matrix = OrderedDict
spec_csr_matrix = {'values': nb.types.int64[:], 'col_index': nb.types.int64[:], 'row_index': nb.types.int64[:]}


@nb.experimental.jitclass(spec_csr_matrix)
class CSRMatrix(object):
    # a minimal csr matrix implementation ala wikipedia
    # used for the backward and forward stars of nodes
    def __init__(self, values, col_index, row_index):
        self.values = values
        self.col_index = col_index
        self.row_index = row_index

    def get_nnz(self, row):
        # getting all the non zero columns of a particular row
        row_start = self.row_index[row]
        row_end = self.row_index[row + 1]
        return self.col_index[row_start:row_end]

    def get_row(self, row):
        row_start = self.row_index[row]
        row_end = self.row_index[row + 1]
        return self.values[row_start:row_end]
