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
from collections import OrderedDict
from heapq import heappop, heappush

import numba as nb
import numpy as np
from numba.core.types import float32, uint8, uint32, uint64

numba_csr_val_types = [float32[:], uint32[:], uint8[:]]


@nb.njit(cache=True)
def csr_sort(index_array, values, number_of_columns):
    """
    sorts index_array increasing according to rows, ties are broken by the columns
    example for sorted index_array:
    array([[0, 1],
           [1, 480640],
           [2, 3],
           [2, 356104], dtype=uint32)
    Parameters
    ----------
    index_array : 2d array with each row containing the indexes of nnz element, uint32
    values : 1d array with corresponding value, any type

    Returns
    -------

    """
    # need to use reflected list here ([]) and not the faster typed List(), see ticket
    # https://github.com/numba/numba/issues/4926
    # function that sorts the index array and value array for the sparse matrix creation
    my_heap = [(np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0))]
    sorted_index_array = np.empty_like(index_array)
    sorted_values = np.empty_like(values)
    for index, edge in enumerate(index_array):
        i, j = edge
        heappush(
            my_heap,
            (
                uint64(i * (number_of_columns + 1) + j),
                uint64(i),
                uint64(j),
                uint64(index),
            ),
        )
    c = 0

    heappop(my_heap)  # removing init val
    while len(my_heap) > 0:
        tuple = heappop(my_heap)
        i, j, index = tuple[1], tuple[2], tuple[3]
        sorted_index_array[c] = uint32(i), uint32(j)
        sorted_values[c] = values[uint32(index)]
        c += 1
    return sorted_index_array, sorted_values


def __build_csr_cls(nb_type):
    """

    Parameters
    ----------
    nb_type : numba array type of value array, e.g. uint32[:] or float32[:]

    Returns
    -------

    """
    spec_csr_matrix = [
        ("values", nb_type),
        ("col_index", nb.types.uint32[:]),
        ("row_index", nb.types.uint32[:]),
        ("nnz_rows", nb.types.uint32[:]),
        ("number_of_rows", uint32),
    ]
    spec_csr_matrix = OrderedDict(spec_csr_matrix)

    @nb.experimental.jitclass(spec_csr_matrix)
    class CSRMatrix(object):
        # a minimal csr matrix implementation a la wikipedia
        # get_nnz and get_row should only be used on rows for which a value is present
        # otherwise IndexErrors will be raised
        def __init__(self, values, col_index, row_index):
            self.values = values
            self.col_index = col_index
            self.row_index = row_index
            self.nnz_rows = self.__set_nnz_rows()
            self.number_of_rows = len(row_index) - 2

        def get_nnz(self, row):
            # getting all the non zero columns of a particular row
            row_start = self.row_index[row]
            row_end = self.row_index[row + 1]
            return self.col_index[row_start:row_end]

        def get_row(self, row):
            row_start = self.row_index[row]
            row_end = self.row_index[row + 1]
            return self.values[row_start:row_end]
            # except Exception:
            #   return np.empty(0, dtype=np_type)

        def __set_nnz_rows(self):
            rows = []
            for row in np.arange(len(self.row_index[:-1]), dtype=np.uint32):
                if len(self.get_nnz(row)) > 0:
                    rows.append(row)
            return np.array(rows, dtype=np.uint32)

        def get_nnz_rows(self):
            # get rows that have non-zero values
            return self.nnz_rows

    return CSRMatrix


@nb.njit(cache=True)
def csr_prep(index_array, values, shape, unsorted=True):
    """

    Parameters
    ----------
    index_array :  2d array with each row containing the indexes of nnz element, uint32
    values : 1d array with corresponding value, any type
    shape : shape of sparse matrix (rows,colums), uint32/64
    unsorted : index_array and values sorted or not, see csr_sort, boolean

    Returns
    -------

    """
    if np.max(index_array[:, 1]) > (shape[1] - 1) or np.max(index_array[:, 0]) > (
        shape[0] - 1
    ):
        raise ValueError(
            "dimensions are smaller than respective cols and rows in index array"
        )
    if unsorted:
        index_array, values = csr_sort(index_array, values, shape[1])
    col, row = _csr_format(index_array, shape[0])
    return values, col, row


@nb.njit(cache=True)
def _csr_format(index_array, number_of_rows):
    """

    Parameters
    ----------
    index_array : 2d array with each row containing the indexes of nnz element
    number_of_rows : size of sparse matrix
    Returns
    -------

    """
    # index_array with the position of the elements (i,j),
    # i being the row and j the column
    # sorted by rows with ties settled by column.
    # Values sorted accordingly, see csr_sort
    col, row = nb.typed.List.empty_list(item_type=uint32), nb.typed.List.empty_list(
        item_type=uint32
    )
    row.append(np.uint32(0))
    row_value_counter = np.uint32(0)
    processed_edges = np.uint32(0)
    empty_csr = len(index_array) == 0
    for i in np.arange(number_of_rows + 1, dtype=np.uint32):
        edges_in_row = np.uint32(0)
        if empty_csr:
            row.append(row_value_counter)
        else:
            for edge in index_array[processed_edges:]:
                if i == edge[0]:
                    col.append(
                        index_array[np.uint32(processed_edges + edges_in_row), 1]
                    )
                    row_value_counter += np.uint32(1)
                    edges_in_row += np.uint32(1)
                else:
                    # next row
                    row.append(row_value_counter)
                    processed_edges += edges_in_row
                    break

    return np.asarray(col, dtype=np.uint32), np.asarray(row, dtype=np.uint32)


F32CSRMatrix = __build_csr_cls(nb.float32[:])
UI32CSRMatrix = __build_csr_cls(nb.uint32[:])
UI8CSRMatrix = __build_csr_cls(nb.uint8[:])

try:
    ui32csr_type = UI32CSRMatrix.class_type.instance_type
    f32csr_type = F32CSRMatrix.class_type.instance_type
    ui8csr_type = UI8CSRMatrix.class_type.instance_type
except Exception:
    # numba disabled
    ui32csr_type = None
    f32csr_type = None
    ui8csr_type = None

# empty initilization below
# col, row = _csr_format(np.array([[]]), 4)
# val = np.array([], dtype=np.float32)
# my_csr = F32CSRMatrix(val, col, row)
