#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
# file containing all the parameters used in the assignment procedures for DTA
# ids for event handling
network_loading_method='iltm'
link_capacity_id=2
registered_events=[True, False]
max_capacity=10000
v_wave_default=30 # 30 km/h as default backward wave speed
turn_capacity_default=9999
turn_type_default=0
turn_t0_default=0
node_capacity_default=9999
node_control_default=0

#0: capacity
config_dict = {
               'traffic_keys': {'capacity', 'length', 'maxspeed', 'flow'}
               }
visualization_keys_edges = ['capacity', 'length', 'maxspeed', 'flow', 'name', 'travel_time', 'osmid'
    , 'compressed_osm_ids', 'time','_id','lanes']
visualization_keys_nodes = ['osmid', 'originating_traffic', 'destination_traffic', 'x', 'y','_id']
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
assignment_method_defaults = {'DUE': 'bpr,flow_avg', 'SUN':'dial_uncongested','SUE':'dial_congested'}
number_of_control_arrays=0
ramp_metering_control_array_index=1
log_to_file = True
log_folder = 'logs'
log_level = 20
log_filename = 'dtapy'