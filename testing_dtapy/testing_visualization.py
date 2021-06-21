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
from dtapy.visualization import show_dynamic_network, show_demand, show_network
from dtapy.demand import generate_od_xy, parse_demand
from dtapy.visualization import xt_plot

rand_speed = np.random.rand(100, 100) * 120
xt_plot(data_array=rand_speed, detector_locations=[2, 4, 8], X=10, T=1, title='xt_plot_test', type='speed')
g: nx.MultiDiGraph = load_pickle(city + '_grid_centroids')
g.number_of_edges()
time = SimulationTime(0, 2, 0.25)
# show random link cost and flow state in a network
flows = np.random.rand(time.tot_time_steps * g.number_of_edges()).reshape(
    (time.tot_time_steps, g.number_of_edges())) * 1200
costs = np.random.rand(time.tot_time_steps * g.number_of_edges()).reshape(
    (time.tot_time_steps, g.number_of_edges())) * 200
convergence = np.arange(1, 0, -0.01)
# names with spaces in them do not work!
link_kwargs = { 'other_flows': flows * 0.7, 'custom_length': np.arange(g.number_of_edges())}
node_kwargs = {
    'random_node_property': np.random.rand(g.number_of_nodes() * time.tot_time_steps * 3).reshape(time.tot_time_steps,
                                                                                                  g.number_of_nodes(),
                                                                                                  3)}

show_dynamic_network(g, time, flows, link_kwargs=link_kwargs, node_kwargs=node_kwargs, convergence=convergence,
                     highlight_links=[1, 7, 26], highlight_nodes=[1, 34, 3])
show_network(g, highlight_nodes=np.arange(10), highlight_links=np.arange(10),
             node_kwargs={'random_node_property': np.random.rand(g.number_of_nodes())})

# visualize random demand for a single time period
_json = generate_od_xy(170, city, seed=0)
demand_graph = parse_demand(_json, g, time=0)
show_demand(demand_graph)
