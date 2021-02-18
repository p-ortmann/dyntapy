#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
# file containing all the parameters used in the assignment procedures for DTA
# ids for event handling
# TODO: Switch to named tuples as shown below
# from collections import namedtuple
# add docs as shown below!!
import numpy as np
from typing import NamedTuple


class Parameters(NamedTuple):
    """container for categories of parameters"""
    visualization: NamedTuple
    supply: NamedTuple
    demand: NamedTuple
    network_loading: NamedTuple
    route_choice: NamedTuple


class Supply(NamedTuple):
    """
    container for parameters that concern supply
    """
    max_capacity: np.uint32
    v_wave_default: np.float32


class Demand(NamedTuple):
    """
    container for parameters that concern demand
    """
    default_connector_speed: np.float32
    default_connector_capacity: np.float32
    default_connector_lanes: np.uint8


class NetworkLoading(NamedTuple):
    """
    container for parameters that concern supply
    """
    method: str
    gap: np.float32
    max_iterations: np.uint32


class Visualization(NamedTuple):
    """
    container for parameters that concern visualization
    """
    max_links: int


class RouteChoice(NamedTuple):
    """
    container for parameters that concern route choice
    """
    aggregation: np.float32
    delta_t: np.float32


# supply_categories = namedtuple('supply_categories', 'max_capacity v_wave_default turn_capacity_default turn_type_default'
#                                                     ' turn_t0_default node_capacity_default node_control_default')
# supply = None
#
#
# parameter_categories = namedtuple('parameter_categories', 'visualization network_loading route_choice supply demand')
# parameters =  None
from math import pow
import numpy as np

network_loading_method = 'iltm'
link_capacity_id = 2
registered_events = [True, False]
max_capacity = 10000
v_wave_default = 30  # 30 km/h as default backward wave speed
turn_capacity_default = 9999
turn_type_default = 0
turn_t0_default = 0
node_capacity_default = 9999
node_control_default = 0
route_choice_agg = 1
route_choice_dt = 1
route_choice_delta =  0.01


i_ltm_gap = np.float32(pow(10, -7))
i_ltm_max_it = np.uint32(5000)
ltm_dt = 0.25
# 0: capacity
config_dict = {
    'traffic_keys': {'capacity', 'length', 'maxspeed', 'flow'}
}
#TODO: remove this filter - all should be shown ..
visualization_keys_edges = ['capacity', 'length', 'maxspeed', 'flow', 'name', 'travel_time', 'osmid'
    , 'compressed_osm_ids', 'time', '_id', 'lanes']
visualization_keys_nodes = ['osmid', 'originating_traffic', 'destination_traffic', 'x', 'y', '_id']
default_capacity = 1000
default_speed = 30
cap_mapping = {'motorway': 2000, 'motorway_link': 1500, 'trunk': 1000, 'trunk_link': 1000, 'primary': 1000,
               'primary_link': 1000, 'secondary': 1000, 'secondary_link': 1000, 'tertiary': 600, 'minor': 600,
               'unclassified': 600,
               'residential': 600, 'living_street': 300}
speed_mapping = {'motorway': 100, 'motorway_link': 60, 'trunk': 50, 'trunk_link': 50, 'primary': 50,
                 'primary_link': 50, 'secondary': 50, 'secondary_link': 50, 'tertiary': 30, 'minor': 30,
                 'unclassified': 30,
                 'residential': 30, 'living_street': 15}
max_links_visualized = 6000
assignment_method_defaults = {'DUE': 'bpr,flow_avg', 'SUN': 'dial_uncongested', 'SUE': 'dial_congested'}
number_of_control_arrays = 0
ramp_metering_control_array_index = 1
log_to_file = True
log_folder = 'logs'
log_level = 20
log_filename = 'dtapy'
LTM_GAP = np.float32(pow(10, 8))
