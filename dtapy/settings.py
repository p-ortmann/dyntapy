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

# File IO and Logging

log_to_file = True
log_folder = 'logs'
log_level = 20
log_numba = False  # this will affect performance dramatically, and should only be set to true for debugging

default_city = 'Zinnowitz'


# Parameters for various methods
@dataclass
class _Supply:
    """
    container for parameters that concern supply
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
    default_connector_speed: np.float32 = np.float32(1000)
    default_connector_capacity: np.float32 = np.float32(10000)
    default_connector_lanes: np.uint8 = np.uint8(10)
    default_centroid_spacing: int = 500  # in meters


@dataclass
class _NetworkLoading:
    """
    container for parameters that concern supply
    """
    link_model: str = 'i_ltm'
    gap: np.float32 = np.float32(0.001)
    max_iterations: np.uint32 = np.uint32(100)
    step_size: np.float32 = np.float32(0.25)
    node_model: str = 'orca'


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
    link_width_scaling = 0.012
    node_size = 6
    link_highlight_color = '#ff6ec7'  # neon pink
    node_highlight_color = '#ff6ec7'  # neon pink
    node_color = '#424949'  # dark gray
    centroid_color = '#9063CD'  # purple
    link_color = ' #f2f4f4' #light gray,  only for no flow scenarios


@dataclass
class _RouteChoice:
    """
    container for parameters that concern route choice
    """
    aggregation: np.float32 = 1
    step_size: np.float32 = 0.25  # time discretization
    delta_cost: np.float32 = 0.01


@dataclass
class _Parameters:
    """container for categories of parameters"""
    visualization: _Visualization = _Visualization()
    supply: _Supply = _Supply()
    demand: _Demand = _Demand()
    network_loading: _NetworkLoading = _NetworkLoading()
    route_choice: _RouteChoice = _RouteChoice()


parameters = _Parameters()  # default initilisation
