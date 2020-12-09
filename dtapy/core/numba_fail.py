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
from os import environ
spec = [('arr', nb.types.int64[:])]

@nb.experimental.jitclass(spec)
class MyClass(object):
    def __init__(self, arr):
        self.arr = arr

    def get_row(self, row):
        return self.arr[row]


arr = np.arange(10, dtype=np.int64)
my_inst = MyClass(arr)
print(my_inst.get_row(12))
