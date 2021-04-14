#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
#
import numpy as np

# import node model you want to test
from dtapy.core.network_loading.node_models.orca_nodel_model import orca_node_model as node_model


# defining the inputs, example taken from Chris MJ, et al. "A generic class of first order node models for
#     dynamic macroscopic simulation of traffic flows." Transportation Research Part B:
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
turn_capacity = np.full(16, 2000, dtype=np.float32)  # set to link capacity to avoid the results differing
# from the paper example
turning_flow = np.empty_like(turning_fractions)
for in_link, flow in enumerate(sending_flow):
    turning_flow[in_link] = flow * turning_fractions[in_link]

results = node_model(sending_flow, turning_fractions, turning_flow, receiving_flow, turn_capacity,
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
    print(f'differences are {results - correct_turning_flows} with {results} and {correct_turning_flows}')
