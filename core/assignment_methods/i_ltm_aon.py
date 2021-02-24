#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from assignment import Assignment
from core import setup_aon
from core import calc_turning_fractions
def i_ltm_aon(assignment:Assignment):
    aon_state = setup_aon(assignment)
    calc_turning_fractions(assignment, aon_state)
