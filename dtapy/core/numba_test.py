#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
# test file to check whether something works in jitclass or jitted function..
import numba as nb
import numpy as np
spec = [('arr', nb.types.int64[:])]

@nb.experimental.jitclass(spec)
class MyClass(object):
    def __init__(self):
        pass

    def set_arr(self, arr):
        self.arr=arr


arr = np.arange(10, dtype=np.int64)
my_inst = MyClass()
my_inst.set_arr(arr)

