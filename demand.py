#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import osmnx as ox
from osmnx.distance import great_circle_vec, euclidean_dist_vec
import geopandas as gpd
from shapely.geometry import Point
from geojson import Feature, FeatureCollection, dumps
import networkx as nx
import geojson
from shapely.geometry import LineString
from collections import deque
from json import loads
import itertools
from settings import parameters
from network_data import sort_graph
from utilities import log

default_connector_speed = parameters.demand.default_connector_speed
default_connector_capacity = parameters.demand.default_connector_capacity
default_connector_lanes = parameters.demand.default_connector_lanes
default_centroid_spacing = parameters.demand.default_centroid_spacing


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
        g[u][v]['maxspeed'] = default_connector_speed
        g[u][v]['capacity'] = default_connector_capacity
        g[u][v]['lanes'] = default_connector_lanes


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


def od_graph_from_matrix(od_matrix: np.ndarray, X, Y):
    """
    creates od_graph from od_matrix and centroid locations
    Parameters
    ----------
    od_matrix : float array, centroids x centroids
    X : lon array of centroids
    Y : lat array of centroids

    Returns
    -------
    od_flow_graph: nx.DiGraph
    """
    pass
    # TODO: add this functionality


def get_centroid_grid_coords(name: str, spacing=default_centroid_spacing):
    """
    creates centroids on a grid that overlap with the polygon that is associated with city or region specified
    under 'name'
    Parameters
    ----------
    name : name of the city to be used as reference polygon
    spacing : distance between two adjacent centroids on the grid

    Returns
    -------
    X,Y arrays of centroid locations
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
    x,y = np.array(centroids.x), np.array(centroids.y)
    log(f'found {x.size} centroids at {default_centroid_spacing=} meter', to_console=True)
    return x,y


def add_centroids_to_graph(g, X, Y, k=1, add_connectors=True, sort=True):
    """
    adds centroids to g as the first C-1 nodes, with C the number of centroids.
    g.nodes.c contains 'x_coord','y_coord' and 'centroid' with x and y coords as given in X,Y  and 'centroid' a boolean
    set to True.

    Parameters
    ----------
    sort : boolean, whether to sort the nodes of the graph, see sort_graph(g)
    add_connectors : whether to add auto-configured connectors, k for each centroid
    Y : lat vector of centroids
    X : lon vector of centroids
    k : number of connectors to be added per centroid
    g : nx.MultiDigraph containing only road network edges and nodes with 'x_coord' and 'y_coord' attribute on each node
    Returns
    -------
    new nx.MultiDiGraph with centroids (and connectors)
    """
    new_g = nx.MultiDiGraph()
    new_g.graph = g.graph
    if len([u for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict]) > 0:
        raise ValueError('grid generation assumes that no centroids are present in the graph')
    new_centroids = [(u, {'x_coord': p[0], 'y_coord': p[1], 'centroid': True, 'node_id': u}) for u, p in
                     enumerate(zip(X, Y))]
    connector_id = itertools.count()
    # the steps below here could be compressed but not without compromising the consistency between the order in
    # edges and nodes in the networkx graph and the 'link_id' and 'node_id' attributes
    for u, data in new_centroids:  # first centroids, then intersection nodes for order
        new_g.add_node(u, **data)
    for u, data in g.nodes.data():
        new_g.add_node(u, **data)
    for u, v, data in g.edges.data():
        new_g.add_edge(u, v, **data)
    if add_connectors:
        for u, data in new_centroids:
            tmp: nx.DiGraph = g  # calculate distance to road network graph
            og_nodes = list(g.nodes)
            for _ in range(k):
                # find the nearest node j k times, ignoring previously nearest nodes in consequent iterations if
                # multiple connectors are wanted
                v, length = get_nearest_node(tmp, (data['y_coord'], data['x_coord']), return_dist=True)
                og_nodes.remove(v)
                tmp = tmp.subgraph(og_nodes)
                source_data = {'connector': True, 'length': length / 1000, 'free_speed': default_connector_speed,
                               'lanes': default_connector_lanes,
                               'capacity': default_connector_capacity, 'link_id': next(connector_id),
                               'link_type': np.int8(1), 'from_node_id': u, 'to_node_id': v}  # length in km
                sink_data = {'connector': True, 'length': length / 1000, 'free_speed': default_connector_speed,
                             'lanes': default_connector_lanes,
                             'capacity': default_connector_capacity, 'link_id': next(connector_id),
                             'link_type': np.int8(-1), 'from_node_id': v, 'to_node_id': u}
                new_g.add_edge(u, v, **source_data)
                new_g.add_edge(v, u, **sink_data)
    if sort:
        new_g = sort_graph(new_g)
    return new_g


def parse_demand(data: str, g: nx.DiGraph, time=0):
    """
    Maps travel demand to existing closest centroids in g.
    The demand pattern is expressed as its own directed graph od_graph returned with 'time', 'crs' and 'name' as metadata
    in od_graph.graph : Dict.
    The od_graph contains edges with a 'flow' entry that indicates the movements from centroid to centroid.
    The corresponding OD table can be retrieved through calling nx.to_scipy_sparse_matrix(od_graph,weight='flow' )

    Parameters
    ----------
    time : time stamp for the demand data in hours (0<= time <=24)
    data : geojson that contains lineStrings (WGS84) as features, each line has an associated
    'flow' stored in the properties dict
    g : nx.MultiDigraph for the city under consideration with centroids assumed to be labelled starting from 0, .. ,C-1
    with C being the number of centroids.

    There's no checking on whether the data and the nx.Digraph correspond to the same geo-coded region.
    Returns
    -------
    od_graph: nx.DiGraph

    """
    od_graph = nx.MultiDiGraph()
    od_graph.graph['time'] = time  # time in hours
    od_graph.graph['crs'] = 'epsg:4326'
    name = g.graph['name']
    od_graph.graph['name'] = f'mobility flows in {name} at {time} s'
    od_graph.add_nodes_from([(u, data_dict) for u, data_dict in g.nodes(data=True) if 'centroid' in data_dict])
    if od_graph.number_of_nodes() == 0:
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
                                                  od_graph)  # snapped centroids are in nx node id space,
    # and not their respective internal centroid ids
    flows = np.array(gdf['flow'])

    ods = snapped_centroids.reshape((int(snapped_centroids.size / 2), 2), order='F')
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

    return od_graph


class DynamicDemand:
    def __init__(self, trip_graphs):
        """

        Parameters
        ----------
        trip_graphs : Dict of nx.DiGraphs with mobility patterns for different time slots t, with t as dict key
        """
        self.trip_graphs = trip_graphs
        self.insertion_times = list(trip_graphs.keys())

    def get_sparse_repr(self, time):
        """
        Parameters
        ----------
        time : integer, time slice to be retrieved

        Returns
        -------
        lil_matrix of trip table for given time slice
        """
        graph: nx.DiGraph = self.trip_graphs[time]
        return nx.to_scipy_sparse_matrix(graph, weight='flow', format='lil')

    def get_trip_graph(self, time):
        return self.trip_graphs[time]


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
        {"x": nx.get_node_attributes(centroid_graph, "x_coord"), "y": nx.get_node_attributes(centroid_graph, "y_coord")}
    )
    tree = cKDTree(data=centroids[["x", "y"]], compact_nodes=True, balanced_tree=True)
    # ox.get_nearest_nodes()
    # query the tree for nearest node to each origin
    points = np.array([X, Y]).T
    centroid_dists, centroid_idx = tree.query(points, k=1)
    snapped_centroids = centroids.iloc[centroid_idx].index
    return np.array(snapped_centroids, dtype=np.uint32), centroid_dists


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


def get_nearest_node(G, point, method="haversine", return_dist=False):
    """
    adapted from OSMNX
    Find the nearest node to a point.

    Return the graph node nearest to some (lat, lng) or (y, x) point and
    optionally the distance between the node and the point. This function can
    use either the haversine formula or Euclidean distance.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : tuple
        The (lat, lng) or (y, x) point for which we will find the nearest node
        in the graph
    method : string {'haversine', 'euclidean'}
        Which method to use for calculating distances to find nearest node.
        If 'haversine', graph nodes' coordinates must be in units of decimal
        degrees. If 'euclidean', graph nodes' coordinates must be projected.
    return_dist : bool
        Optionally also return the distance (in meters if haversine, or graph
        node coordinate units if euclidean) between the point and the nearest
        node

    Returns
    -------
    int or tuple of (int, float)
        Nearest node ID or optionally a tuple of (node ID, dist), where dist
        is the distance (in meters if haversine, or graph node coordinate
        units if euclidean) between the point and nearest node
    """
    if len(G) < 1:
        raise ValueError("G must contain at least one node")

    # dump graph node coordinates into a pandas dataframe indexed by node id
    # with x and y columns
    coords = ((n, d["x_coord"], d["y_coord"]) for n, d in G.nodes(data=True))
    df = pd.DataFrame(coords, columns=["node", "x", "y"]).set_index("node")

    # add columns to df for the (constant) coordinates of reference point
    df["ref_y"] = point[0]
    df["ref_x"] = point[1]

    # calculate the distance between each node and the reference point
    if method == "haversine":
        # calculate distances using haversine for spherical lat-lng geometries
        dists = great_circle_vec(lat1=df["ref_y"], lng1=df["ref_x"], lat2=df["y"], lng2=df["x"])

    elif method == "euclidean":
        # calculate distances using euclid's formula for projected geometries
        dists = euclidean_dist_vec(y1=df["ref_y"], x1=df["ref_x"], y2=df["y"], x2=df["x"])

    else:
        raise ValueError('method argument must be either "haversine" or "euclidean"')

    # nearest node's ID is the index label of the minimum distance
    nearest_node = dists.idxmin()

    # if caller requested return_dist, return distance between the point and the
    # nearest node as well
    if return_dist:
        return nearest_node, dists.loc[nearest_node]
    else:
        return nearest_node
