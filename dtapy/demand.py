# This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from stapy.setup import float_dtype, int_dtype
from numba.typed import List, Dict
from shapely.geometry import Point
from collections import namedtuple
import pandas as pd
import numpy as np
from dtapy.core.jitclasses import SimulationTime, StaticDemand, DynamicDemand
from datastructures.csr import csr_prep, F32CSRMatrix
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from geojson import Feature, FeatureCollection, dumps
import networkx as nx
import geojson
from shapely.geometry import LineString
from scipy.sparse import lil_matrix, csr, csr_matrix
import itertools
from collections import deque


def build_demand_structs(od_matrix):
    demand_dict = Dict()
    od_flow_vector = []
    origins = set(od_matrix.nonzero()[0].astype(str(int_dtype)))
    for i in origins:
        my_list = List()
        destinations = od_matrix.getrow(i).tocoo().col.astype(str(int_dtype))
        demands = od_matrix.getrow(i).tocoo().data.astype(str(int_dtype))
        # discarding intrazonal traffic ..
        origin_index = np.where(destinations == i)
        destinations = np.delete(destinations, origin_index)
        demands = np.delete(demands, origin_index)
        assert len(destinations) == len(demands)
        for demand in demands:
            od_flow_vector.append(float_dtype(demand))
        my_list.append(destinations)
        my_list.append(demands)
        demand_dict[i] = my_list
    od_flow_vector = np.array(od_flow_vector)
    return demand_dict, od_flow_vector


def generate_od_fixed(number_of_nodes, number_of_od_values, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes, dtype=np.uint32)
    vals = np.random.choice(arr, size=number_of_od_values, replace=False)
    ids = [np.where(arr.reshape((number_of_nodes, number_of_nodes)) == val) for val in vals]
    for i, j in ids:
        i, j = int(i), int(j)
        if isinstance(rand_od.rows[i], list):
            rand_od.rows[i].append(j)
            rand_od.data[i].append(int(np.random.random() * 2000))
        else:
            rand_od.rows[i] = list(j)
            rand_od.data[i] = list((int(np.random.random() * 2000)))
    return rand_od


def generate_random_bush(number_of_nodes, number_of_branches, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=str(int_dtype))
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes)
    origin = np.random.randint(0, number_of_nodes)
    destinations = np.random.choice(np.arange(0, number_of_nodes), number_of_branches, replace=False)
    for destination in destinations:
        i, j = int(origin), int(destination)
        if isinstance(rand_od.rows[i], list):
            rand_od.rows[i].append(j)
            rand_od.data[i].append(int(np.random.random() * 2000))
        else:
            rand_od.rows[i] = list(j)
            rand_od.data[i] = list((int(np.random.random() * 2000)))
    return rand_od


def generate_od_xy(tot_ods, name: str, max_flow=2000):
    """

    Parameters
    ----------
    tot_ods :total number of OD pairs to be generated
    name : str, name of the city or region to geocode and sample from
    max_flow : maximum demand that is generated

    Returns
    -------
    geojson containing lineStrings and flows
    """
    # 4326 : WGS84
    # 3857 : web mercator
    my_gdf: gpd.geodataframe.GeoDataFrame = ox.geocode_to_gdf(name)
    tot_points = 2000 * tot_ods
    my_gdf.to_crs(3857, inplace=True)
    X = np.random.random(tot_points) * (my_gdf.bbox_east[0] - my_gdf.bbox_west[0]) + my_gdf.bbox_west[0]
    # np.normal.normal means uniform distribution between 0 and 1, can easily be replaced (gumbel, gaussian..)
    Y = np.random.random(tot_points) * (my_gdf.bbox_north[0] - my_gdf.bbox_south[0]) + my_gdf.bbox_south[0]
    points = [Point(x, y) for x, y in zip(X, Y)]
    my_points = gpd.geoseries.GeoSeries(points, crs=4326)
    my_points = my_points.to_crs(3857)
    valid_points = my_points[my_points.within(my_gdf.loc[0, 'geometry'])].to_crs(4326)
    X = np.array(valid_points.geometry.x[:tot_ods * 2])
    Y = np.array(valid_points.geometry.y[:tot_ods * 2])
    destinations = [(x, y) for x, y in zip(X[:tot_ods], Y[:tot_ods])]
    origins = [(x, y) for x, y in zip(X[tot_ods:], Y[tot_ods:])]
    vals = np.random.random(tot_ods) * max_flow
    my_features = []
    line_strings = [LineString([[origin[0], origin[1]], [destination[0], destination[1]]]) for origin, destination in
                    zip(origins, destinations)]
    tmp = [{'flow': f} for f in vals]
    my_features = [Feature(geometry=my_linestring, properties=my_tmp) for my_linestring, my_tmp in
                   zip(line_strings, tmp)]
    fc = FeatureCollection(my_features)

    return dumps(fc)


def _check_centroid_connectivity(g: nx.DiGraph):
    centroids = [u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict]
    tot_out_c = [__count_iter_items(g.successors(n)) for n in centroids]
    tot_in_c = [__count_iter_items(g.predecessors(n)) for n in centroids]
    if min(tot_out_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_out_c) if val == 0]
        raise ValueError(f'these centroids do not have an outgoing connector: {disconnected_centroids}')
    if min(tot_in_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_in_c) if val == 0]
        raise ValueError(f'these centroids do not have an incoming connector: {disconnected_centroids}')


def create_centroids(data: str, g: nx.DiGraph, matching_dist=200, k=3, return_od=True, time=0):
    """
    adds connectors and centroids to g based on demand in geojson format.
    k connectors are added for each centroid, connecting them to the k closest network nodes.
    centroids that are already present in g are 'snapped to' under the matching distance threshold.
    The demand pattern is added in the graph as its own directed graph and can be retrieved via g.graph['od_graph'],
     it contains edges with a 'flow' entry that indicates the movements from centroid to centroid.

    Parameters
    ----------
    time : time stamp for the demand data in seconds, can be used as UNIX epoch to specify dates.
    data : geojson that contains lineStrings (WGS84) as features, each line has an associated
    'flow' stored in the properties dict
    g : networkx DiGraph for the city under consideration, assumes set up as shown in the network_data files
    matching_dist : distance in meters under which centroids are merged into one another, the centroid which is retained
     depends on the order in which they were added to the graph
    k : int, number of connectors to be added for each centroid
    return_od : True if you want the OD matrix returned in lil matrix.

    There's no checking on whether the data and the nx.Digraph correspond to the same geo-coded region.
    Returns
    -------
    lil_matrix od table if return_od is set to True.

    Note that you can add nodes as fixed centroids before calling this function and set the matching distance
    to infinity or some large number to ensure that demand is always mapped to existing centroids.

    """
    centroid_graph = g.subgraph(nodes=[u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict])
    if centroid_graph.number_of_nodes() > 0:
        _check_centroid_connectivity(g)
    data = geojson.loads(data)
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    X0 = [gdf.geometry[u].xy[0][0] for u in range(len(gdf))]
    X1 = [gdf.geometry[u].xy[0][1] for u in range(len(gdf))]
    Y0 = [gdf.geometry[u].xy[1][0] for u in range(len(gdf))]
    Y1 = [gdf.geometry[u].xy[1][1] for u in range(len(gdf))]
    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1))
    snapped_centroids, dists = find_nearest_centroids(X,
                                                      Y,
                                                      centroid_graph)  # snapped centroids are in nx node id space, and not their
    # respective internal centroid ids
    tot_ods = len(X1)
    new_centroids = []
    u = max(g.nodes)
    new_centroid_ids = np.empty(tot_ods * 2, dtype=np.int64)

    c = centroid_graph.number_of_nodes() - 1  # assumed to be labelled consecutively from 0 to c-1
    for x, y, dist, snapped_centroid, i in zip(X, Y, dists, snapped_centroids,
                                               range(len(snapped_centroids))):

        if dist > matching_dist:
            c = c + 1
            u = u + 1
            new_centroids.append((u, {'x': x, 'y': y, 'centroid': True, 'centroid_id': c}))
            new_centroid_ids[i] = c
        else:
            new_centroid_ids[i] = centroid_graph.nodes[snapped_centroid]['centroid_id']
    road_network = g.subgraph([u for u, data_dict in g.nodes(data=True) if 'centroid' not in data_dict])
    for u, data in new_centroids:
        g.add_node(u, **data)
        tmp: nx.DiGraph = road_network
        og_nodes = list(road_network.nodes)
        for _ in range(k):
            v, length = ox.get_nearest_node(tmp, (data['y'], data['x']), return_dist=True)
            og_nodes.remove(v)
            tmp = tmp.subgraph(og_nodes)
            connector_data = {'connector': True, 'length': length}
            g.add_edge(u, v, **connector_data)
            g.add_edge(v, u, **connector_data)
    flows = np.array(gdf['flow'])
    centroid_graph = g.subgraph(nodes=[u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict]).copy()
    number_of_centroids = max(new_centroid_ids) + 1
    O, D = np.array_split(new_centroid_ids, 2)
    od_edges = [(u, v, {'flow': flow}) for u, v, flow in zip(O, D, flows)]
    centroid_graph.add_edges_from(od_edges)
    centroid_graph.graph['time'] = np.array([time])
    g.graph['od_graph'] = centroid_graph
    if return_od:
        if len(time) == 1:
            return csr_matrix((flows, (O, D)), shape=(number_of_centroids, number_of_centroids)).tolil()
        else:
            lil_matrices = []
            for i in range(len(time)):
                flow = flows[:][i]
                lil_matrices.append(csr_matrix((flow[flow != 0], (O[flow != 0], D[flow != 0])),
                                               shape=(number_of_centroids, number_of_centroids)).tolil())




def build_demand(demand_data, insertion_times, simulation_time: SimulationTime, number_of_nodes):
    """
    
    Parameters
    ----------
    simulation_time : time object, see class def
    demand_data : List <scipy.lil_matrix> # node x node demand, each element is added demand for a particular moment in time
    insertion_times : Array, times at which the demand is loaded

    Returns
    -------

    """
    # internally we maintain different labels for origins and destinations starting at zero,
    # the corresponding node ids are stored in the static demand object
    # this makes it easier to work with destination based flows ( keeping the label space small)

    if not np.all(insertion_times[1:] - insertion_times[:-1] > simulation_time.step_size):
        raise ValueError('insertion times are assumed to be monotonously increasing. The minimum difference between '
                         'two '
                         'insertions is the internal simulation time step')
    times = np.arange(simulation_time.start, simulation_time.end, simulation_time.step_size)
    loading_time_steps = [(np.abs(insertion_time - times)).argmin() for insertion_time in insertion_times]
    static_demands = List()
    rows = [np.asarray(lil_demand.nonzero()[0], dtype=np.uint32) for lil_demand in demand_data]
    row_sizes = np.array([lil_demand.nonzero()[0].size for lil_demand in demand_data], dtype=np.uint32)
    cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
    col_sizes = np.array([lil_demand.nonzero()[1].size for lil_demand in demand_data], dtype=np.uint32)
    all_destinations, cols = np.unique(np.concatenate(cols), return_inverse=True)
    all_origins, rows = np.unique(np.concatenate(rows), return_inverse=True)
    cols = np.array_split(cols, np.cumsum(col_sizes))
    rows = np.array_split(rows, np.cumsum(row_sizes))
    tot_destinations = all_destinations.size
    tot_origins = all_origins.size
    row_counter = 0
    col_counter = 0

    for internal_time, lil_demand, row, col in zip(loading_time_steps, demand_data, rows, cols):
        vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
        index_array_to_d = np.column_stack((row, col))
        index_array_to_o = np.column_stack((col, row))
        to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (number_of_nodes, number_of_nodes)))
        to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (number_of_nodes, number_of_nodes)))
        origin_node_ids = np.array([all_origins[i] for i in to_destinations.get_nnz_rows()], dtype=np.uint32)
        destination_node_ids = np.array([all_destinations[i] for i in to_origins.get_nnz_rows()], dtype=np.uint32)
        static_demands.append(StaticDemand(to_origins, to_destinations,
                                           to_origins.get_nnz_rows(), to_destinations.get_nnz_rows(), origin_node_ids,
                                           destination_node_ids, internal_time))

    return DynamicDemand(static_demands, simulation_time.tot_time_steps, all_origins, all_destinations)


def __count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(itertools.izip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def find_nearest_centroids(X, Y, centroid_graph: nx.DiGraph):
    """

    Parameters
    ----------
    data : geojson file with OD flow graph
    centroid_graph : nx.DiGraph with existing centroids

    Returns
    -------

    """

    tot_ods = len(X)
    centroids = pd.DataFrame(
        {"x": nx.get_node_attributes(centroid_graph, "x"), "y": nx.get_node_attributes(centroid_graph, "y")}
    )
    tree = cKDTree(data=centroids[["x", "y"]], compact_nodes=True, balanced_tree=True)

    # query the tree for nearest node to each origin
    points = np.array([X, Y]).T
    centroid_dists, centroid_idx = tree.query(points, k=1)
    try:
        snapped_centroids_idx = centroids.iloc[centroid_idx].index

    except IndexError:
        assert centroid_graph.number_of_nodes() == 0
        snapped_centroids = np.full(tot_ods, -1)  # never accessed, only triggered if centroids is empty.
        # - all distances are inf
    return snapped_centroids, centroid_dists
