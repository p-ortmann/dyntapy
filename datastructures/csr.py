#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
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
from numba.core.types import float64, int64

valid_csr_val_types = [float64[:], int64[:]]

def csr_sorted(func):
    # decorator that sorts the index array and value array for the sparse matrix creation
    # np.lexsort is not implemented in numba
    @functools.wraps(func)
    def with_sorted_inputs(index_array=None, values=None, number_of_rows=None):
        seq = np.lexsort((index_array[:, 1], index_array[:, 0]))
        index_array = [index_array[ind] for ind in seq]
        values = [values[ind] for ind in seq]
        index_array = np.array(index_array)
        values = np.array(values)
        return func(index_array, values, number_of_rows)

    return with_sorted_inputs

def __build_csr_cls():
    cls = []
    for my_type in valid_csr_val_types:
        spec_csr_matrix = OrderedDict
        spec_csr_matrix = {'values': my_type, 'col_index': nb.types.int64[:], 'row_index': nb.types.int64[:]}

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

        cls.append(CSRMatrix)
    return cls
def __build_csr_funs(cls):
    funs=[]
    for cl, type in cls, valid_csr_val_types:
        @csr_sorted
        @nb.njit
        def construct_sparse_matrix(index_array, values, number_of_rows):
            """

            Parameters
            ----------
            index_array : 2d array with each row containing the indexes of nnz element
            values : values in order of index array
            number_of_rows :
            val_type : type of the values

            Returns
            -------

            """

            # index_array with the position of the elements (i,j), i being the row and j the column
            # sorted by rows with ties settled by column. Values sorted accordingly, see csr_sorted decorator
            # example for valid index_array:
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
            return cl(np.asarray(values, type), np.asarray(col, dtype=np.int64),
                                np.asarray(row, dtype=np.int64))
        funs.append(construct_sparse_matrix)
    return funs




cls = __build_csr_cls()
F64CSRMatrix, I64CSRMatrix=__build_csr_cls()
construct_sparse_matrix_i64, construct_sparse_matrix_f64 = __build_csr_funs([F64CSRMatrix, I64CSRMatrix])








