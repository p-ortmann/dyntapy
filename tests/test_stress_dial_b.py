from numba import config
config.DISABLE_JIT = 1

import os
import pathlib
import sys
from pickle import dump, load

one_up = pathlib.Path(__file__).parents[1]
sys.path.append(one_up.as_posix())

import numpy as np
from dyntapy.settings import parameters

parameters.static_assignment.dial_b_cost_differences = 0.00001
from dyntapy.demand_data import generate_od_xy, add_centroids, \
    auto_configured_centroids, parse_demand
from dyntapy import StaticAssignment
from dyntapy import show_network
from dyntapy.supply_data import road_network_from_place, relabel_graph
from dyntapy.sta._debugging_sta import loading, continuity

# this file stress-tests the DIAL B implementation with more demand, a bigger network
# and multiple connectors per centroid.
method = 'constant'  # 'constant' or 'iterated' if multiple demand scenarios
# should be explored, you can run this until failure to identify seeds that trigger
# an error.
# stress tested for epsilon up to 0.00001, not lower
#
seed_constant = 13
tot_seeds_to_try = 50
tot_od_pairs = 10
max_flow_per_od_pair = 500


# should be run to explore if errors would occur under different queueing conditions

#@mark.skip(reason='too computationally expensive')
def test_stress_dial_b():
    city = 'Stralsund'
    HERE = pathlib.Path(__file__).parent
    one_up = pathlib.Path(__file__).parents[1]
    sys.path.append(one_up.as_posix())
    file_path_network = HERE.as_posix() + os.path.sep + city.lower() + '_road_network'
    try:
        with open(file_path_network, 'rb') as my_file:
            g = load(my_file)
        show_network(g)
    except FileNotFoundError:

        g = road_network_from_place(city, buffer_dist_close=5000,
                                    buffer_dist_extended=20000)
        print('road network graph acquired')
        x, y, names, place_tags = auto_configured_centroids(city,
                                                            buffer_dist_close=5000,
                                                            buffer_dist_extended=10000)
        print('centroids found')
        g = add_centroids(g, x, y, k=3, method='turn',
                          name=names, place=place_tags)
        print('centroids added to graph')
        g, inverse = relabel_graph(g, return_inverse=True)
        with open(file_path_network, 'wb') as network_file:
            dump(g, network_file)
            print(f'network saved at f{file_path_network}')
        show_network(g)

    if method == 'iterated':
        # we iterate
        for seed in range(tot_seeds_to_try):
            print(f'testing for demand from {seed=}')
            json_demand = generate_od_xy(tot_od_pairs, city, seed=seed,
                                         max_flow=max_flow_per_od_pair)
            od_graph = parse_demand(json_demand, g)
            assignment = StaticAssignment(g, od_graph)
            results = assignment.run(method='dial_b')
            print(f'test for demand from {seed=} passed successfully')
            loading_ok, _, _ = loading(assignment.internal_demand,
                                       assignment.internal_network,
                                       results.flows)
            assert loading_ok
            continuity_violations, _, _ = continuity(flows=results.flows,
                                                     network=assignment.internal_network)
            assert not np.any(continuity_violations)

    elif method == 'constant':
        print(f'testing for demand from {seed_constant=}')
        json_demand = generate_od_xy(tot_od_pairs, city, seed=seed_constant,
                                     max_flow=max_flow_per_od_pair)
        od_graph = parse_demand(json_demand, g)
        assignment = StaticAssignment(g, od_graph)
        results = assignment.run(method='dial_b')
        show_network(g,flows=results.flows, highlight_links=[891,839])
        loading_failed, _, _ = loading(assignment.internal_demand,
                                   assignment.internal_network,
                                   results.flows)
        assert not loading_failed
        continuity_violations, _, _ = continuity(flows=results.flows,
                                                 network=assignment.internal_network,
                                                 numerical_threshold=0.0001)
        assert not np.any(continuity_violations)


    print('dial passed successfully')


if __name__ == '__main__':
    test_stress_dial_b()
