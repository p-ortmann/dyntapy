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
@nb.njit
def casting():
    my_list=nb.typed.List()
    for i in np.arange(10, dtype=nb.int64):
        my_list.append(i)
    return np.asarray(my_list)