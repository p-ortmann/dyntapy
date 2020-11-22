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
import numba as nb
import numpy as np
from collections import OrderedDict
import functools


def csr_sorted(func):
    # decorator that sorts the index array and value array for the sparse matrix creation
    # handled in this way because np.lexsort is not implemented in numba
    @functools.wraps(func)
    def with_sorted_inputs(index_array=None, values=None, number_of_rows=None):
        seq = np.lexsort((index_array[:, 1], index_array[:, 0]))
        index_array = [index_array[ind] for ind in seq]
        values = [values[ind] for ind in seq]
        index_array = np.array(index_array, dtype=np.int64)
        values = np.array(values, dtype=np.int64)
        return func(index_array, values, number_of_rows)

    return with_sorted_inputs


@csr_sorted
@nb.njit
def construct_sparse_link_matrix(index_array, values, number_of_rows):
    # index_array with the position of the elements (i,j), i being the row and j the column
    # sorted by rows with ties settled by column. Values sorted accordingly, see csr_sorted decorator
    # example:
    # array([[0, 1],
    #        [1, 480640],
    #        [2, 3],
    #        [2, 356104], dtype=int64)
    col, row = nb.typed.List(), nb.typed.List()
    row.append(0)
    row_counter = 0
    link_counter = 0
    link_id = 0
    for i in np.arange(number_of_rows + 1, dtype=np.int64):
        link_counter = 0
        for edge in index_array[link_id:]:
            if i == edge[0]:
                # values.append(link_id+link_counter)
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
    # a minimal csr matrix implementation a la wikipedia
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
