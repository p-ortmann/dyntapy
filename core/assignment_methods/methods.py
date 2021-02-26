#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dataclasses import dataclass
from typing import Callable
from core.assignment_methods.i_ltm_aon import i_ltm_aon


@dataclass
class valid_methods:
    i_ltm_aon: Callable = i_ltm_aon
