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
log_numba = False  # this will affect performance dramatically, and should only be set to true for debugging

numba_config = {  # see https://numba.pydata.org/numba-doc/dev/reference/envvars.html for config
    # DISABLE_JIT doesn't work for now as jit-classes are not supported in pure python ..
    # List creation from arrays is also not correct with jit disabled,
    # see https://github.com/numba/numba/issues/6001
    'NUMBA_CACHE_DIR': os.getcwd() + os.path.sep + 'numba_cache',
    'NUMBA_DEBUG': '0',
    'NUMBA_DEBUG_CACHE': '0',
    'NUMBA_DEVELOPER_MODE': '0',
    'NUMBA_FULL_TRACEBACKS': '0',
}
default_city = 'Zinnowitz'


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
    default_connector_speed: np.float32 = np.float32(1000)
    default_connector_capacity: np.float32 = np.float32(10000)
    default_connector_lanes: np.uint8 = np.uint8(10)
    default_centroid_spacing: int = 2000  # in meters


@dataclass
class _NetworkLoading:
    """
    container for parameters that concern supply,
    Note: Changes to these values may only take affect after the numba_cache directory has been deleted
    """
    link_model: str = 'i_ltm'  # only 'i_ltm' for now
    gap: np.float32 = np.float32(0.001)
    epsilon: np.float = np.float32(0.001)
    max_iterations: np.uint32 = np.uint32(1000)
    step_size: np.float32 = np.float32(0.25)
    node_model: str = 'orca'  # 'only 'orca' for now
    precision: np.float32 = np.float32(0.001)


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
    link_color = ' #f2f4f4'  # light gray,  only for no flow scenarios


@dataclass
class _RouteChoice:
    """
    container for parameters that concern route choice
    Note: Changes to these values may only take affect after the numba_cache directory has been deleted
    """
    aggregation: np.float32 = 0.5
    step_size: np.float32 = 0.25  # time discretization
    delta_cost: np.float32 = 0.001 # in hours


class _Assignment:
    """
        container for parameters that concern the actual assignment, e.g. how route choice and network loading
        are wired up with each other.
        Note: Changes to these values may only take affect after the numba_cache directory has been deleted
        """
    gap: np.float32 = 0.0001
    max_iterations: np.uint = 3
    smooth_turning_fractions: str = 'MSA'  # valid entry only 'MSA' for now
    smooth_costs: bool = False


@dataclass
class _Parameters:
    """container for categories of parameters"""
    visualization: _Visualization = _Visualization()
    supply: _Supply = _Supply()
    demand: _Demand = _Demand()
    network_loading: _NetworkLoading = _NetworkLoading()
    route_choice: _RouteChoice = _RouteChoice()
    assignment: _Assignment = _Assignment()


parameters = _Parameters()  # default initilisation
