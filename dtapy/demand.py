# This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from scipy.spatial import cKDTree
from numba.typed import List
import pandas as pd
import numpy as np
from dtapy.core.assignment_cls import SimulationTime, StaticDemand, DynamicDemand
from datastructures.csr import csr_prep, F32CSRMatrix
import osmnx as ox
from osmnx.distance import great_circle_vec
import geopandas as gpd
from shapely.geometry import Point
from geojson import Feature, FeatureCollection, dumps
import networkx as nx
import geojson
from shapely.geometry import LineString
from collections import deque
from json import loads
import itertools
from dtapy.parameters import parameters

DEFAULT_CONNECTOR_SPEED = parameters.demand.default_connector_speed
DEFAULT_CONNECTOR_CAPACITY = parameters.demand.default_connector_capacity
DEFAULT_CONNECTOR_LANES = parameters.demand.default_connector_lanes


def generate_od_xy(tot_ods, name: str, max_flow=2000, seed=0):
    """

    Parameters
    ----------
    seed : numpy random seed
    tot_ods :total number of OD pairs to be generated
    name : str, name of the city or region to geocode and sample from
    max_flow : maximum demand that is generated

    Returns
    -------
    geojson containing lineStrings and flows
    """
    # 4326 : WGS84
    # 3857 : web mercator
    np.random.seed(seed)
    my_gdf: gpd.geodataframe.GeoDataFrame = ox.geocode_to_gdf(name)
    tot_points = 20 * tot_ods  # could be improved by using the area ratio between bbox and polygon for scaling
    X = np.random.random(tot_points) * (my_gdf.bbox_east[0] - my_gdf.bbox_west[0]) + my_gdf.bbox_west[0]
    # np.normal.normal means uniform distribution between 0 and 1, can easily be replaced (gumbel, gaussian..)
    Y = np.random.random(tot_points) * (my_gdf.bbox_north[0] - my_gdf.bbox_south[0]) + my_gdf.bbox_south[0]
    points = [Point(x, y) for x, y in zip(X, Y)]
    my_points = gpd.geoseries.GeoSeries(points, crs=4326)
    valid_points = my_points[my_points.within(my_gdf.loc[0, 'geometry'])]  # bounding box typically doesn't align
    # with polygon extend so we ought to check which points are inside
    X = np.array(valid_points.geometry.x[:tot_ods * 2])
    Y = np.array(valid_points.geometry.y[:tot_ods * 2])
    destinations = [(x, y) for x, y in zip(X[:tot_ods], Y[:tot_ods])]
    origins = [(x, y) for x, y in zip(X[tot_ods:], Y[tot_ods:])]
    vals = np.random.random(tot_ods) * max_flow
    line_strings = [LineString([[origin[0], origin[1]], [destination[0], destination[1]]]) for origin, destination in
                    zip(origins, destinations)]
    tmp = [{'flow': f} for f in vals]
    my_features = [Feature(geometry=my_linestring, properties=my_tmp) for my_linestring, my_tmp in
                   zip(line_strings, tmp)]
    fc = FeatureCollection(my_features)
    return dumps(fc)


def __set_connector_default_attr(g):
    connectors = [(u, v) for u, v, data in g.edges.data() if 'connector' in data]
    connector_sg = nx.edge_subgraph(g, connectors)
    for edge in connector_sg.edges:
        u, v = edge
        g[u][v]['maxspeed'] = DEFAULT_CONNECTOR_SPEED
        g[u][v]['capacity'] = DEFAULT_CONNECTOR_CAPACITY
        g[u][v]['lanes'] = DEFAULT_CONNECTOR_LANES


def _check_centroid_connectivity(g: nx.DiGraph):
    """
    verifies if each centroid has at least one connector
    Parameters
    ----------
    g : nx.Digraph

    Returns
    -------

    """
    centroids = [u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict]
    tot_out_c = [__count_iter_items(g.successors(n)) for n in centroids]
    tot_in_c = [__count_iter_items(g.predecessors(n)) for n in centroids]
    if min(tot_out_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_out_c) if val == 0]
        raise ValueError(f'these centroids do not have an outgoing connector: {disconnected_centroids}')
    if min(tot_in_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_in_c) if val == 0]
        raise ValueError(f'these centroids do not have an incoming connector: {disconnected_centroids}')


def __create_centroid_grid(name: str, spacing=1000):
    """
    creates centroids on a grid that overlap with the polygon that is associated with city or region specified
    under 'name'
    Parameters
    ----------
    name : name of the city to be used as reference polygon
    spacing : distance between two adjacent centroids on the grid

    Returns
    -------

    """
    my_gdf = ox.geocode_to_gdf(name)
    range_ns_meters = great_circle_vec(my_gdf.bbox_north[0], my_gdf.bbox_east[0], my_gdf.bbox_south[0],
                                       my_gdf.bbox_east[0])
    range_ew_meters = great_circle_vec(my_gdf.bbox_east[0], my_gdf.bbox_south[0], my_gdf.bbox_west[0],
                                       my_gdf.bbox_south[0])
    ns_tics = np.linspace(my_gdf.bbox_south[0], my_gdf.bbox_north[0], np.int(np.floor(range_ns_meters / spacing)))
    ew_tics = np.linspace(my_gdf.bbox_west[0], my_gdf.bbox_east[0], np.int(np.floor(range_ew_meters / spacing)))
    grid = np.meshgrid(ew_tics, ns_tics)
    X = grid[0].flatten()
    Y = grid[1].flatten()
    points = [Point(x, y) for x, y in zip(X, Y)]
    points = gpd.geoseries.GeoSeries(points, crs=4326)
    centroids = points[points.within(my_gdf.loc[0, 'geometry'])]
    return centroids


def add_centroids_from_grid(name: str, g, D=2000, k=3):
    """
    partitions the polygon associated with the region/city into squares (with D as the side length in meters)
    and adds one centroid and k connectors to the k nearest nodes for each square.
    Parameters
    ----------
    k : number of connectors to be added per centroid
    g : nx.Digraph for name generated by osmnx
    D : side length of squares
    name : name of the city to which g corresponds
    geojson : geojson string, containing Points which are either origins or destinations.

    Returns
    -------

    """
    if len([u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict]) > 0:
        raise ValueError('grid generation assumes that no centroids are present in the graph')
    u0 = max(g.nodes) + 1
    centroids = __create_centroid_grid(name, D)
    new_centroids = [(u, {'x': p.x, 'y': p.y, 'centroid': True, 'centroid_id': c}) for u, p, c in
                     zip(range(u0, u0 + len(centroids)), centroids, range(len(centroids)))]
    for u, data in new_centroids:
        g.add_node(u, **data)
        tmp: nx.DiGraph = g
        og_nodes = list(g.nodes)
        for _ in range(k):
            v, length = ox.get_nearest_node(tmp, (data['y'], data['x']), return_dist=True)
            og_nodes.remove(v)
            tmp = tmp.subgraph(og_nodes)
            connector_data = {'connector': True, 'length': length / 1000}  # length in km
            g.add_edge(u, v, **connector_data)
            g.add_edge(v, u, **connector_data)
    __set_connector_default_attr(g)


def parse_demand(data: str, g: nx.DiGraph, time=0):
    """
    Maps travel demand to existing closest centroids in g.
    The demand pattern is added in the graph as its own directed graph and can be retrieved via g.graph['od_graph'],
    it contains edges with a 'flow' entry that indicates the movements from centroid to centroid.
    The corresponding OD tables can be retrieved through calling nx.to_scipy_sparse_matrix(od_graph,weight='flow' )

    Parameters
    ----------
    time : time stamp for the demand data in seconds, can be used as UNIX epoch to specify dates.
    data : geojson that contains lineStrings (WGS84) as features, each line has an associated
    'flow' stored in the properties dict
    g : networkx DiGraph for the city under consideration with centroids

    There's no checking on whether the data and the nx.Digraph correspond to the same geo-coded region.
    Returns
    -------


    """
    centroid_subgraph = g.subgraph(nodes=[u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict])
    if centroid_subgraph.number_of_nodes() == 0:
        raise ValueError('Graph does not contain any centroids.')
    data = geojson.loads(data)
    gdf = gpd.GeoDataFrame.from_features(data['features'])
    X0 = [gdf.geometry[u].xy[0][0] for u in range(len(gdf))]
    X1 = [gdf.geometry[u].xy[0][1] for u in range(len(gdf))]
    Y0 = [gdf.geometry[u].xy[1][0] for u in range(len(gdf))]
    Y1 = [gdf.geometry[u].xy[1][1] for u in range(len(gdf))]
    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1))
    snapped_centroids, _ = find_nearest_centroids(X,
                                                  Y,
                                                  centroid_subgraph)  # snapped centroids are in nx node id space,
    # and not their respective internal centroid ids
    new_centroid_ids = np.array([centroid_subgraph.nodes[u]['centroid_id'] for u in snapped_centroids], dtype=np.uint32)
    flows = np.array(gdf['flow'])
    od_graph = nx.DiGraph()
    od_graph.add_nodes_from(
        [(data['centroid_id'], {'nx_id': u, 'x': data['x'], 'y': data['y']}) for u, data in g.nodes(data=True) if
         'centroid' in data])
    ods = new_centroid_ids.reshape((int(new_centroid_ids.size / 2), 2), order='F')
    uniq, inv, counts = np.unique(ods, axis=0, return_inverse=True, return_counts=True)

    if uniq.shape[0] != ods.shape[0]:
        new_flows = []
        for val, c, i in zip(uniq, counts, inv):
            if c > 1:
                idx = np.where(np.all(ods == val, axis=1))
                new_flows.append(np.sum(flows[idx]))
                #  amalgamate duplicates
            else:
                new_flows.append(flows[i])
        ods = uniq

    od_edges = [(od[0], od[1], {'flow': flow}) for od, flow in zip(ods, flows)]
    od_graph.add_edges_from(od_edges)
    od_graph.graph['time'] = time
    od_graph.graph['crs'] = 'epsg:4326'
    name = g.graph['name']
    od_graph.graph['name'] = f'mobility flows in {name} at {time} s'
    if 'od_graph' in g.graph:
        if type(g.graph['od_graph']) != list:
            raise ValueError('od_graph needs to be a list')
        g.graph['od_graph'].append(od_graph)
    else:
        g.graph['od_graph'] = [od_graph]


def _build_demand(demand_data, insertion_times, simulation_time: SimulationTime):
    """
    
    Parameters
    ----------
    simulation_time : time object, see class def
    demand_data : List <scipy.lil_matrix> origins x destinations,
     matrix k corresponds to the demand inserted at time[k]
    insertion_times: Array, times at which the demand is loaded assumed to be in hours

    Returns
    -------

    """
    if not np.all(insertion_times[1:] - insertion_times[:-1] > simulation_time.step_size):
        raise ValueError('insertion times are assumed to be monotonously increasing. The minimum difference between '
                         'two '
                         'insertions is the internal simulation time step')
    if max(insertion_times > 24):
        raise ValueError('internally time is restricted to 24 hours')
    time = np.arange(simulation_time.start, simulation_time.end, simulation_time.step_size)
    loading_time_steps = [(np.abs(insertion_time - time)).argmin() for insertion_time in insertion_times]
    static_demands = List()
    rows = [np.asarray(lil_demand.nonzero()[0], dtype=np.uint32) for lil_demand in demand_data]
    row_sizes = np.array([lil_demand.nonzero()[0].size for lil_demand in demand_data], dtype=np.uint32)
    cols = [np.asarray(lil_demand.nonzero()[1], dtype=np.uint32) for lil_demand in demand_data]
    col_sizes = np.array([lil_demand.nonzero()[1].size for lil_demand in demand_data], dtype=np.uint32)
    all_destinations, cols = np.unique(np.concatenate(cols), return_inverse=True)
    all_origins, rows = np.unique(np.concatenate(rows), return_inverse=True)
    cols = np.array_split(cols, np.cumsum(col_sizes))
    rows = np.array_split(rows, np.cumsum(row_sizes))
    tot_destinations = max(all_destinations)
    tot_origins = max(all_origins)
    for internal_time, lil_demand, row, col in zip(loading_time_steps, demand_data, rows, cols):
        vals = np.asarray(lil_demand.tocsr().data, dtype=np.float32)
        index_array_to_d = np.column_stack((row, col))
        index_array_to_o = np.column_stack((col, row))
        to_destinations = F32CSRMatrix(*csr_prep(index_array_to_d, vals, (tot_origins, tot_destinations)))
        to_origins = F32CSRMatrix(*csr_prep(index_array_to_o, vals, (tot_origins, tot_destinations)))
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
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def find_nearest_centroids(X, Y, centroid_graph: nx.DiGraph):
    """
    Parameters
    ----------
    X : longitude of points epsg 4326
    Y : latitude of points epsg 4326
    centroid_graph : nx.DiGraph with existing centroids, coordinates stored as 'x' and 'y' in epsg 4326

    Returns
    -------

   """
    if centroid_graph.number_of_nodes() == 0:
        raise ValueError('graph does not contain any centroids')
    tot_ods = len(X)
    assert centroid_graph.graph['crs'] == 'epsg:4326'
    centroids = pd.DataFrame(
        {"x": nx.get_node_attributes(centroid_graph, "x"), "y": nx.get_node_attributes(centroid_graph, "y")}
    )
    tree = cKDTree(data=centroids[["x", "y"]], compact_nodes=True, balanced_tree=True)
    # ox.get_nearest_nodes()
    # query the tree for nearest node to each origin
    points = np.array([X, Y]).T
    centroid_dists, centroid_idx = tree.query(points, k=1)
    snapped_centroids = centroids.iloc[centroid_idx].index
    return snapped_centroids, centroid_dists


def _merge_gjsons(geojsons):
    """

    Parameters
    ----------
    geojsons : List of geojson Strings

    Returns
    -------
    merged geojson dict with all features

    """
    feature_lists = [loads(my_string)['features'] for my_string in geojsons]
    features = list(itertools.chain(*feature_lists))
    return {'type': 'FeatureCollection', 'features': features}
