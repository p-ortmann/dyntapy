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
#
# file containing all the parameters used in the assignment procedures for DTA
import numpy as np
from dataclasses import dataclass
import os

# File IO and Logging

log_to_file = True
log_folder = 'logs'
log_level = 20

# debugging
log_numba = False  # this will affect performance dramatically, and should only be set to true for debugging
debugging = False  # whether to run various consistency checks during the computation

default_dynamic_city = 'Zinnowitz'
default_static_city = 'Leuven'
# static and dynamic defaults differ because the network files have a different supply- and demand interface
# (turn vs link connectors), this avoids name clashing and overwrites in the test files.

# Parameters for various methods
@dataclass
class _Supply:
    """
    container for parameters that concern supply,
    Note: Changes to these values may only take affect after the numba_cache directory has been deleted
    """
    max_capacity: np.uint32 = np.uint32(10000)
    v_wave_default: np.float32 = np.float32(30)
    turn_capacity_default = np.float32(9999)
    turn_type_default = np.uint8(0)
    turn_t0_default = np.float32(0)
    node_capacity_default = np.float32(9999)
    node_control_default = np.uint8(0)
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


@dataclass
class _Demand:
    """
    container for parameters that concern demand
    """
    default_connector_speed: np.float32 = np.float32(200)
    default_connector_capacity: np.float32 = np.float32(10000)
    default_connector_lanes: np.uint8 = np.uint8(10)
    default_centroid_spacing: int = 500  # in meters


@dataclass
class _NetworkLoading:
    """
    container for parameters that concern supply,
    Note: Changes to these values may only take affect after the numba_cache directory has been deleted
    """
    link_model: str = 'i_ltm'  # only 'i_ltm' for now
    gap: np.float32 = np.float32(0.001)
    epsilon: np.float = np.float32(0.001)
    trigger_node_update_threshold  = np.float32(0.01)
    cong_flow_delta_trigger = np.float(0.01) # how much a constraint at the outflow side has to be violated
    # before it's considered congested
    max_iterations: np.uint32 = np.uint32(1000)
    node_model: str = 'orca'  # 'only 'orca' for now
    precision: np.float32 = np.float32(0.001)
    use_turn_delays: np.bool = False  # whether or not to take into account turn delays in the route choice
    # more of a space holder settings as it's not yet taken integrated into propagation ..


@dataclass
class _Visualization:
    """
    container for parameters that concern visualization
    """
    max_links: np.uint32 = np.uint32(10000)
    plot_size: int = 800
    notebook_plot_size: int = 600
    link_keys = ['link_id', 'from_node_id', 'to_node_id', 'length', 'capacity', 'free_speed', 'ext_id',
                 'name', 'facility_type', 'flow', 'connector']
    node_keys = ['node_id', 'x_coord', 'y_coord', 'ext_id', 'node_type', 'ctrl_type', 'centroid']
    link_width_scaling = 0.01
    node_size = 6
    link_highlight_colors = ['#ff6ec7', '#6effef', '#7fff6e', '#6e7fff', '#ffa66e',
                             '#999999']  # neon pink, cyan, lime green
    # light blue, orange, gray
    node_highlight_color = '#ff6ec7'  # neon pink
    node_color = '#424949'  # dark gray
    centroid_color = '#ff00b7'  # purple
    link_color = ' #f2f4f4'  # light gray,  only for no flow scenarios


@dataclass
class _RouteChoice:
    """
    container for parameters that concern route choice
    Note: Changes to these values may only take affect after the numba_cache directory has been deleted
    """
    aggregation: np.float32 = 1
    delta_cost: np.float32 = 0.0001  # in hours
    precision: np.float32 = 0.0001
    restricted_turn_cost = 24


@dataclass
class _Dynamic_Assignment:
    """
        container for parameters that concern the actual assignment, e.g. how route choice and network loading
        are wired up with each other.
        Note: Changes to these values may only take affect after the numba_cache directory has been deleted
        """
    gap: np.float32 = 0.001
    max_iterations: np.uint = 5
    smooth_turning_fractions: str = 'MSA'  # valid entry only 'MSA' for now
    smooth_costs: bool = False


@dataclass
class _Dynamic_Parameters:
    """container for categories of parameters"""
    visualization = _Visualization()
    supply = _Supply()
    demand = _Demand()
    network_loading = _NetworkLoading()
    route_choice = _RouteChoice()
    assignment = _Dynamic_Assignment()


@dataclass
class _Static_Assignment:
    methods = {'DUE': 'bpr,flow_avg', 'SUN': 'dial_uncongested', 'SUE': 'dial_congested'}
    bpr_alpha = np.double(0.15)
    bpr_beta = np.double(4)
    msa_max_iterations = 200
    msa_delta = 0.001
    fw_max_iterations = 50
    fw_delta = 0.001
    dial_b_max_iterations = 30
    dial_b_cost_differences = 0.01
    logit_theta = 0.001
    gap_method = "relative"


@dataclass
class _Static_Parameters:
    visualization = _Visualization()  # if there ever is a need for different settings for
    # static and dynamic on supply/demand/visualization just make a different class for them
    supply = _Supply()
    demand = _Demand()
    assignment = _Static_Assignment()


dynamic_parameters = _Dynamic_Parameters()  # default initilisation
static_parameters = _Static_Parameters()
