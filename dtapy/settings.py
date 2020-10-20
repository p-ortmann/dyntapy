import numpy as np
"""simple parameter file to avoid passing parameters repeatedly """
config_dict = {'bpr_alpha': np.double(0.15),
               'bpr_beta': np.double(4),
               'msa_max_iterations': 10,
               'msa_delta': 0.001,
               'traffic_keys': {'capacity', 'length', 'maxspeed', 'flow'}
               }
visualization_keys_edges = ['capacity', 'length', 'maxspeed', 'flow', 'name', 'travel_time', 'osmid'
    , 'compressed_osm_ids', 'time','_id']
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

assignment_parameters = {'bpr_alpha': np.double(0.15),
                         'bpr_beta': np.double(4),
                         'msa_max_iterations': 200,
                         'msa_delta': 0.001,
                         'fw_max_iterations': 50,
                         'fw_delta': 0.001,
                         'dial_b_cost_differences': 0.001,
                         'logit_theta':0.001,
                         }


gap_method = "relative"

# logging setup
log_to_file = True
log_folder = 'logs'
log_level = 20
log_filename = 'stapy'

