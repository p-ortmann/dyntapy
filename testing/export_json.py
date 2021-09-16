#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dyntapy.json_export import nx_to_geojson
from dyntapy.network_data import load_pickle

g=load_pickle('antwerp_grid_centroids')
nx_to_geojson(g, to_file=True, city_name='Antwerp')