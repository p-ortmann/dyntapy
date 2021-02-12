#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from numba import njit
from dtapy.core.jitclasses import Network, ILTMState, SimulationTime
from dtapy.assignment import Assignment
from dtapy.parameters import route_choice_dt, route_choice_agg
import numpy as np
#TODO: add generic results object
@njit
def dynamic_shortest_path(assignment: Assignment):
    travel_times = assignment.results.costs
    time_step = assignment.time.step_size
    tot_time_steps = assignment.time.tot_time_steps
    rc_dt = route_choice_dt
    rc_agg = route_choice_agg
    tot_destinations = assignment.dynamic_demand.all_destinations
    tot_nodes = assignment.network.tot_nodes
    tot_links = assignment.network.tot_links
    tot_turns = assignment.network.tot_turns
    turning_fractions = np.zeros((tot_time_steps, tot_turns, tot_destinations))




