#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

from dtapy.settings import default_city as city
from dtapy.network_data import load_pickle
import networkx as nx
import numpy as np
from dtapy.core.time import SimulationTime
from dtapy.visualization import show_assignment

g: nx.MultiDiGraph = load_pickle(city + '_grid_centroids')
g.number_of_edges()
time = SimulationTime(0,2,0.25)
flows = np.random.rand(time.tot_time_steps*g.number_of_edges()).reshape((time.tot_time_steps,g.number_of_edges()))*1200
costs = np.random.rand(time.tot_time_steps*g.number_of_edges()).reshape((time.tot_time_steps,g.number_of_edges()))*200
show_assignment(g, flows,costs,time)