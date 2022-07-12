#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from numba import config
# config.DEBUG = 1
# config.DISABLE_JIT=1
import numpy as np
#np.seterr(all='raise')
from test_dyntapy import test_sue

if __name__ == '__main__':
    # test_get_graph()
    #test_dta()
    # test_shortest_path()
    # test_dial_b()
    # test_sun()
    # test_selected_link_analysis()
    test_sue()

