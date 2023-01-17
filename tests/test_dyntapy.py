#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import os
import pathlib
import sys
from pickle import dump, load

import numpy as np

one_up = pathlib.Path(__file__).parents[1]
sys.path.append(one_up.as_posix())

from dyntapy.demand import DynamicDemand
from dyntapy.demand_data import od_graph_from_matrix
from dyntapy.assignments import DynamicAssignment
from dyntapy.dta.orca_nodel_model import orca_node_model
from dyntapy import get_shortest_paths, get_all_shortest_paths, get_toy_network
from dyntapy.demand import SimulationTime
from dyntapy.demand_data import generate_od_xy, add_centroids, \
    auto_configured_centroids, parse_demand
from dyntapy import StaticAssignment
from dyntapy.sta._debugging_sta import loading, continuity
from dyntapy import show_demand, show_network, show_dynamic_network, \
    show_link_od_flows
from dyntapy.results import get_od_flows, get_selected_link_analysis
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.results import StaticResult
from dyntapy import kspwlo_esx

_city = 'Leuven'
graph = None
result: [StaticResult] = None
demand = None
network = None
assignment = None
HERE = pathlib.Path(__file__).parent
one_up = pathlib.Path(__file__).parents[1]
sys.path.append(one_up.as_posix())
file_path_network = HERE.as_posix() + os.path.sep + _city.lower() + '_road_network'
loading_eps = 0.001
continuity_eps = 0.001


def _check_continuity(flows: np.ndarray, method: str):
    continuity_violations, _, _ = continuity(flows=flows,
                                             network=network,
                                             numerical_threshold=continuity_eps)
    loading_failed, _, _ = loading(assignment.internal_demand,
                               assignment.internal_network,
                               flows)
    if np.any(continuity_violations) or loading_failed:
        print(f'continuity issues with {method = }')
        assert not np.any(continuity_violations)
        assert not loading_failed


def test_get_graph(city=_city, k=1, connector_type='turn'):
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
    g = add_centroids(g, x, y, k=k, method=connector_type,
                      name=names, place=place_tags)
    print('centroids added to graph')
    g, inverse = relabel_graph(g, return_inverse=True)
    with open(file_path_network, 'wb') as network_file:
        dump(g, network_file)
        print(f'network saved at f{file_path_network}')
    show_network(g)
    global graph
    graph = g


try:
    with open(file_path_network, 'rb') as my_file:
        graph = load(my_file)
except FileNotFoundError:
    test_get_graph()
seed = 1
json_demand = generate_od_xy(40, _city, seed=seed, max_flow=900)
od_graph = parse_demand(json_demand, graph)

assignment = StaticAssignment(graph, od_graph)
network = assignment.internal_network
demand = assignment.internal_demand


def test_shortest_path():
    dist, paths = get_shortest_paths(graph, 15, [100, 220], return_paths=True)
    print(f'found two {paths} with {dist=}')
    show_network(graph, highlight_links=paths[0] + paths[1],
                 highlight_nodes=[15, 100, 220])
    dist, predecessors = get_all_shortest_paths(graph, 15)
    print(f'one to all passed, acquired {dist=}')


def test_dial_b():
    global result
    method = 'dial_b'
    result = assignment.run(method)
    show_network(graph, flows=result.flows)
    _check_continuity(flows=result.flows, method= method)
    print(f'static assignment method {method=} ran successfully')


def test_kspwlo():
    out_links = network.nodes.out_links
    costs = network.links.length / network.links.free_speed
    is_centroid = network.nodes.is_centroid
    source = 0
    target = 1000
    k = 4
    max_overlap = 0.8
    detour_rejection = 0.3
    paths, distances = kspwlo_esx(costs, out_links, source, target, k, is_centroid,
                                  max_overlap, detour_rejection)
    print(f'found {len(paths)} paths, {distances =} ')
    show_network(graph, highlight_links=paths)


def test_sue():
    global result
    method = 'sue'
    result = assignment.run(method, max_iterations=10, max_gap=0.00001)
    show_network(graph, flows=result.flows)
    _check_continuity(flows=result.flows, method= method)

    print(f'static assignment method {method=} ran successfully')


def test_msa():
    method = 'msa'
    res = assignment.run(method)
    print(f'static assignment method {method=} ran successfully')
    show_network(graph, res.flows)
    _check_continuity(flows=res.flows, method= method)


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
    show_link_od_flows(graph, od_flows)
    sla_od_flows = get_selected_link_analysis(assignment, od_flows, link)
    show_link_od_flows(graph, sla_od_flows, highlight_links=[link])


def test_sun():
    method = 'sun'
    res = assignment.run(method)
    show_network(graph, res.flows)
    _check_continuity(flows=res.flows, method= method)


def test_get_toy_networks():
    for name in [
        "cascetta",
        "simple_bottleneck",
        "simple_merge",
        "simple_diverge",
        "chicagosketch",
        "chicagoregional",
        "siouxfalls",
        "birmingham"]:
        g = get_toy_network(name)
        g = relabel_graph(g)
        show_network(g, euclidean=True)
        print(f'{name=}')


def test_dta():
    g = get_toy_network('cascetta')
    centroid_x = np.array([1, 7, 4])
    centroid_y = np.array([1, 1, 3.5])
    g = add_centroids(g, centroid_x, centroid_y, euclidean=True)
    # also adds connectors automatically
    g = relabel_graph(g)  # adding link and node ids, connectors and centroids
    # are the first elements
    show_network(g, euclidean=True)
    od_matrix = np.zeros(9).reshape((3, 3))
    od_matrix[0, 1] = 500
    od_matrix[2, 1] = 500
    od_graph = od_graph_from_matrix(od_matrix, centroid_x, centroid_y)
    show_demand(od_graph, euclidean=True)
    dynamic_demand = DynamicDemand([od_graph], insertion_times=[0])
    # convert everything to internal representations and parse
    simulation_time = SimulationTime(np.float32(0.0), np.float32(2.0), step_size=0.25)
    assignment = DynamicAssignment(g, dynamic_demand, simulation_time)
    methods = ['incremental_assignment', 'i_ltm_aon']
    for method in methods:
        result = assignment.run(method=method)
    show_dynamic_network(g, simulation_time, flows=result.flows, euclidean=True,
                         link_kwargs={'costs': result.link_costs},
                         )
