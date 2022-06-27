from numba import config
config.DISABLE_JIT = 1

import os
import pathlib
import sys
from pickle import dump, load


one_up = pathlib.Path(__file__).parents[1]
sys.path.append(one_up.as_posix())

from dyntapy.demand_data import generate_od_xy, add_centroids, \
    auto_configured_centroids, parse_demand
from dyntapy import StaticAssignment
from dyntapy import show_network
from dyntapy.supply_data import road_network_from_place, relabel_graph

# this file stress-tests the DIAL B implementation with more demand, a bigger network
# and multiple connectors per centroid.
method = 'constant'  # 'constant' or 'iterated' if multiple demand scenarios
# should be explored, you can run this until failure to identify seeds that trigger
# an error.
seed_constant = 1
tot_od_pairs = 10
max_flow_per_od_pair = 5000
# should be run to explore if errors would occur under different queueing conditions

if __name__ == '__main__':
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
        g = add_centroids(g, x, y, k=3, method='link',
                          name=names, place=place_tags)
        print('centroids added to graph')
        g, inverse = relabel_graph(g, return_inverse=True)
        with open(file_path_network, 'wb') as network_file:
            dump(g, network_file)
            print(f'network saved at f{file_path_network}')
        show_network(g)

    if method == 'iterated':
        # we iterate
        for seed in range(100):
            print(f'testing for demand from {seed=}')
            json_demand = generate_od_xy(tot_od_pairs, city, seed=seed,
                                         max_flow=max_flow_per_od_pair)
            od_graph = parse_demand(json_demand, g)
            assignment = StaticAssignment(g, od_graph)
            results = assignment.run(method='dial_b')
            print(f'test for demand from {seed=} passed successfully')

    elif method == 'constant':
        print(f'testing for demand from {seed_constant=}')
        json_demand = generate_od_xy(tot_od_pairs, city, seed=seed_constant,
                                     max_flow=max_flow_per_od_pair)
        od_graph = parse_demand(json_demand, g)
        assignment = StaticAssignment(g, od_graph)
        results = assignment.run(method='dial_b')

    show_network(g, flows=results.flows)
    print('dial passed successfully')
