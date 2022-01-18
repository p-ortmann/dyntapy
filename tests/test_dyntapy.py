#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import os
import pathlib
from pickle import dump, load

import numpy as np

from dyntapy.dta.orca_nodel_model import orca_node_model
from dyntapy import get_shortest_paths, get_all_shortest_paths
from dyntapy.demand_data import generate_od_xy, add_centroids_to_graph, \
    auto_configured_centroids, parse_demand
from dyntapy import StaticAssignment
from dyntapy import show_demand, show_network, show_dynamic_network, \
    show_link_od_flows
from dyntapy.results import get_od_flows, get_selected_link_analysis
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.results import StaticResult
from tests import cascetta

city = 'Zinnowitz'
graph = None
result: [StaticResult] = None
demand = None
network = None
assignment = None
HERE = pathlib.Path(__file__).parent
file_path_network = HERE.as_posix() + os.path.sep + city.lower() + '_road_network'


def test_get_graph(city=city, k=1, connector_type='link'):
    """

    Parameters
    ----------
    city : str
    k : int, connectors per centroid to be generated
    connector_type: ['turns' , 'links'] whether to add auto-configured
    link-connectors, k*2 for each centroid
    Returns
    -------
    DiGraph
    """
    g = road_network_from_place(city, buffer_dist_close=5000,
                                buffer_dist_extended=20000)
    print('road network graph acquired')
    x, y, names, place_tags = auto_configured_centroids(city, buffer_dist_close=5000,
                                                      buffer_dist_extended=10000)
    print('centroids found')
    g = add_centroids_to_graph(g, x, y, k=k, method=connector_type,
                              name=names, place=place_tags)
    print('centroids added to graph')
    g = relabel_graph(g)
    with open(file_path_network, 'wb') as network_file:
        dump(g, network_file)
        print(f'network saved at f{file_path_network}')
    show_network(g)
    global graph
    graph = g


test_get_graph()
seed = 1
json_demand = generate_od_xy(30, city, seed=seed, max_flow=200)
od_graph = parse_demand(json_demand, graph, 0)


def test_shortest_path():
    dist, paths = get_shortest_paths(graph, 15, [100, 220], return_paths=True)
    print(f'found two {paths} with {dist=}')
    show_network(graph, highlight_links=paths[0] + paths[1],
                 highlight_nodes=[15, 100, 220])
    dist, predecessors = get_all_shortest_paths(graph, 15)
    print(f'one to all passed, acquired {dist=}')


def test_dial_b():
    global assignment
    assignment = StaticAssignment(graph, od_graph)
    global network
    global demand
    global result
    network = assignment.internal_network
    demand = assignment.internal_demand
    method = 'dial_b'
    result = assignment.run(method)
    show_network(graph, result)
    print(f'DUE {method=} ran successfully')


def test_msa():
    loc_assignment = StaticAssignment(graph, od_graph)
    method = 'msa'
    res = loc_assignment.run(method)
    print(f'DUE {method=} ran successfully')
    show_network(graph, res)


def test_node_model():
    # defining the inputs, example taken from Chris MJ,
    # et al. "A generic class of first order node models for
    #     dynamic macroscopic simulation of traffic flows." Transportation Research
    #     Part B:
    #     Methodological 45.1 (2011)

    # sending_flow : array, 1D
    #     turning_fractions : array, dim tot_in_links x tot_out_links
    #     turning_flows : array, dim tot_in_links x tot_out_links
    #     receiving_flow : array, 1D
    #     turn_capacity : array, 1D
    #     in_link_capacity : array, 1D

    sending_flow = np.array([500, 2000, 800, 1700], dtype=np.float32)
    receiving_flow = np.array([1000, 2000, 1000, 2000], dtype=np.float32)
    turning_fractions = np.array([[0, 0.1, 0.3, 0.6],
                                  [0.05, 0, 0.15, 0.8],
                                  [0.125, 0.125, 0, 0.75],
                                  [1 / 17, 8 / 17, 8 / 17, 0]])
    in_link_capacity = np.array([1000, 2000, 1000, 2000], dtype=np.float32)
    turn_capacity = np.full(16, 2000, dtype=np.float32)  # set to link
    # capacity to avoid the results differing
    # from the paper example
    turning_flow = np.empty_like(turning_fractions)
    for in_link, flow in enumerate(sending_flow):
        turning_flow[in_link] = flow * turning_fractions[in_link]

    results = orca_node_model(0, sending_flow, turning_fractions, turning_flow,
                              receiving_flow, turn_capacity,
                              in_link_capacity, len(sending_flow), len(receiving_flow))
    rounded_results = np.round(results, decimals=1)

    correct_turning_flows = np.array([[0, 50, 150, 300],
                                      [68.5, 0, 205.5, 1095.7],
                                      [100, 100, 0, 600],
                                      [80.6, 644.5, 644.5, 0]], dtype=np.float32)

    if np.sum(np.abs(rounded_results - correct_turning_flows)) < 0.0001:
        print('node model test passed successfully')
    else:
        print('node model returns erroneous results ..')
        print(f'differences are {results - correct_turning_flows} '
              f'with {results} and {correct_turning_flows}')


def test_selected_link_analysis():
    link = 622
    od_flows = get_od_flows(assignment, result)
    show_link_od_flows(graph, od_flows, result=result)
    sla_od_flows = get_selected_link_analysis(assignment, od_flows, link)
    show_link_od_flows(graph, sla_od_flows, highlight_links=[link])


def test_sun():
    loc_assignment = StaticAssignment(graph, od_graph)
    method = 'sun'
    res = loc_assignment.run(method)
    show_network(graph, res)


def test_dta():
    cascetta.run()
