#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from numba import njit
import numpy as np
my_a=np.arange(100)

@njit
def delete_val1(my_a, val):
    for index, item in enumerate(my_a):
        if item == val:
            my_a = np.delete(my_a, index)
            break
    return my_a
@njit
def delete_val2(my_a, val):
    for index, item in np.ndenumerate(my_a):
        if item == val:
            my_a = np.delete(my_a, index)
            break
    return my_a
@njit
def delete_val3(my_a, val):
    index= np.where(my_a==val)
    my_a = np.delete(my_a, index[0])
    return my_a
delete_val1(my_a, 62)
delete_val2(my_a,62)
delete_val3(my_a, 62)
#%timeit delete_val1(my_a, 62)
#%timeit delete_val2(my_a, 62)
#%timeit delete_val3(my_a, 62)