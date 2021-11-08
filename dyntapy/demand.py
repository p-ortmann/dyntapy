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
from scipy.sparse import lil_matrix
from numba.typed import List, Dict
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
from dyntapy.settings import dynamic_parameters
from dyntapy.utilities import log

default_connector_speed = dynamic_parameters.demand.default_connector_speed
default_connector_capacity = dynamic_parameters.demand.default_connector_capacity
default_connector_lanes = dynamic_parameters.demand.default_connector_lanes
default_centroid_spacing = dynamic_parameters.demand.default_centroid_spacing


def generate_od_graph(tot_ods, name, g, time, max_flow, seed=0):
    """

    Parameters
    ----------
    tot_ods : total number of OD pairs to be generated
    name : str, name of the city or region to geocode and sample from
    g : nx.MultiDiGraph
    time : time step to which this demand corresponds
    max_flow : maximum demand per pair of points
    seed : numpy random seed

    Returns
    -------

    """
    json = generate_od_xy(tot_ods, name, max_flow, seed)
    return parse_demand(json, g, time)


def generate_od_xy(tot_ods, name: str, max_flow=2000, seed=0):
    """

    Parameters
    ----------
    seed : numpy random seed
    tot_ods :total number of OD pairs to be generated
    name : str, name of the city or region to geocode and sample from
    max_flow : maximum demand per pair of points

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
    creates od_graph(s) from od_matrix and centroid locations, if
    Parameters
    ----------
    od_matrix : float array, centroids x centroids
    X : x_coord array of centroids
    Y : y_coord array of centroids

    Returns
    -------
    od_flow_graph: nx.DiGraph
    """
    if od_matrix.shape != (len(X), len(X)):
        raise ValueError('dimensions of centroid locations and OD matrix incompatible')
    g = nx.MultiDiGraph()
    nodes = [(u, {'x_coord': p[0], 'y_coord': p[1]}) for u, p in enumerate(zip(X, Y))]
    g.add_nodes_from(nodes)
    edges = [(u, v, 0, {'flow': od_matrix[u, v]}) for u, v in np.argwhere(od_matrix > 0) if u!= v ]
    g.add_edges_from(edges)
    return g


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
    x, y = np.array(centroids.x), np.array(centroids.y)
    log(f'found {x.size} centroids at {spacing=} meter', to_console=True)
    return x, y


def add_centroids_to_graph(g, X, Y, k=1, method='turn', euclidean=False):
    """
    Adds centroids to g.
    Each centroids data dict contains 'x_coord','y_coord' and 'centroid' with x and y coords as given in X,Y
    and 'centroid' a boolean set to True.
    if method =='link':
        k*2 connectors are added per centroid, one for each direction as defined below.
    if method == 'turn':
        k*2+2 connectors are added per centroid. There is another artificial node between the centroid and the first
        intersection node. All connector turns share the first starting link from centroid to this artificial node.
    Default attributes for connectors can be
    changed in the settings.


    Parameters
    ----------
    euclidean : bool, toy networks use the euclidean coordinate system
    method : ['turns' , 'links'] whether to add auto-configured link-connectors, k*2 for each centroid
    or auto-configured turn-connectors, k*2+2 for each centroid
    Y : lat vector of centroids
    X : lon vector of centroids
    k : number of road network nodes to connect to per centroid.
    g : nx.MultiDigraph containing only road network edges and nodes with 'x_coord' and 'y_coord' attribute on each node
    Returns
    -------
    new nx.MultiDiGraph with centroids (and connectors)
    """

    new_g = nx.MultiDiGraph()
    new_g.graph = g.graph
    last_intersection_node = max(g.nodes)
    new_centroids = [(u + last_intersection_node + 1, {'x_coord': p[0], 'y_coord': p[1], 'centroid': True}) for u, p in
                     enumerate(zip(X, Y))]
    for u, data in new_centroids:  # first centroids, then intersection nodes for order
        new_g.add_node(u, **data)
    for u, data in g.nodes.data():
        new_g.add_node(u, **data)
    for u, v, data in g.edges.data():
        new_g.add_edge(u, v, **data)
    if method not in ['link', 'turn']:
        raise ValueError
    if method == 'link':
        for u, data in new_centroids:
            add_connectors(data['x_coord'], data['y_coord'], u, k, g, new_g, euclidean)

    if method == 'turn':
        j0 = max(new_g.nodes) + 1
        for j, (u, data) in enumerate(new_centroids):
            if euclidean:
                delta_y = 0.1  # assuming that toy networks work with unit lengths
            else:
                delta_y = (
                                  100 / 6378137) * 180 / np.pi  # placing the artificial connecting node approx. 100 meters north
            new_g.add_node(j0 + j, **{'x_coord': data['x_coord'], 'y_coord': data['y_coord'] + delta_y})
            new_g.add_edge(u, j0 + j, **{'connector': True, 'length': delta_y, 'free_speed': default_connector_speed,
                                         'lanes': default_connector_lanes,
                                         'capacity': default_connector_capacity,
                                         'link_type': np.int8(1)})
            new_g.add_edge(j0 + j, u, **{'connector': True, 'length': delta_y, 'free_speed': default_connector_speed,
                                         'lanes': default_connector_lanes,
                                         'capacity': default_connector_capacity,
                                         'link_type': np.int8(-1)})
            add_connectors(data['x_coord'], data['y_coord'] + delta_y, j0 + j, k, g, new_g, euclidean)

    return new_g


def add_connectors(x, y, u, k, g, new_g, euclidean):
    """
    adding connectors to new_g from starting node u with coordinates x and y to nearest k intersection nodes in g.
    Parameters
    ----------
    euclidean : bool, whether x and y are euclidean
    x : lon
    y : lat
    u : connector from_node in new_g
    k : number of (bidirectional) connectors to add
    g : nx.MultiDiGraph containing all road network nodes
    new_g : nx.MultiDiGraph containing at least u and all nodes of g

    Returns
    -------

    """
    tmp: nx.MultiDiGraph = g  # calculate distance to road network graph
    og_nodes = list(g.nodes)
    for _ in range(k):
        # find the nearest node j k times, ignoring previously nearest nodes in consequent iterations if
        # multiple connectors are wanted
        if not euclidean:
            v, length = get_nearest_node(tmp, (y, x), return_dist=True)
        else:
            v, length = get_nearest_node(tmp, (y, x), method='euclidean',
                                         return_dist=True)
        og_nodes.remove(v)
        tmp = tmp.subgraph(og_nodes)
        source_data = {'connector': True, 'length': length / 1000, 'free_speed': default_connector_speed,
                       'lanes': default_connector_lanes,
                       'capacity': default_connector_capacity,
                       'link_type': np.int8(1)}  # length in km
        sink_data = {'connector': True, 'length': length / 1000, 'free_speed': default_connector_speed,
                     'lanes': default_connector_lanes,
                     'capacity': default_connector_capacity,
                     'link_type': np.int8(-1)}
        new_g.add_edge(u, v, **source_data)
        new_g.add_edge(v, u, **sink_data)


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
    def __init__(self, od_graphs: list, insertion_times):
        """
        multiple edges for same od are not supported right now, MultiDiGraph is chosen to
        maintain compatibility with OSMNX tools
        Parameters
        ----------
        od_graphs :List of nx.MultiDiGraphs
        insertion_times: corresponding times for the demand to be loaded into the network
        """
        if type(od_graphs) is not list:
            raise ValueError
        for item in od_graphs:
            if type(item) is not nx.MultiDiGraph:
                raise ValueError
        self.od_graphs = od_graphs
        self.insertion_times = np.array(insertion_times)

    def get_sparse_repr(self, time):
        """
        Parameters
        ----------
        time : float or integer, time slice to be retrieved

        Returns
        -------
        lil_matrix of trip table for given time slice
        """
        graph = self.get_od_graph(time)
        return nx.to_scipy_sparse_matrix(graph, weight='flow', format='lil')

    def get_od_graph(self, time):
        _id = np.argwhere(self.insertion_times == time)[0][0]
        graph: nx.DiGraph = self.od_graphs[_id]
        return graph


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


########################### STATIC just copied##########################


def build_demand_structs(od_matrix):
    demand_dict = Dict()
    od_flow_vector = []
    origins = set(od_matrix.nonzero()[0].astype(np.uint32))
    for i in origins:
        my_list = List()
        destinations = od_matrix.getrow(i).tocoo().col.astype(np.uint32)
        demands = od_matrix.getrow(i).tocoo().data.astype(np.uint32)
        # discarding intrazonal traffic ..
        origin_index = np.where(destinations == i)
        destinations = np.delete(destinations, origin_index)
        demands = np.delete(demands, origin_index)
        assert len(destinations) == len(demands)
        for demand in demands:
            od_flow_vector.append(np.float32(demand))
        my_list.append(destinations)
        my_list.append(demands)
        demand_dict[i] = my_list
    od_flow_vector = np.array(od_flow_vector)
    return demand_dict, od_flow_vector


def generate_od(number_of_nodes, origins_to_nodes_ratio, origins_to_destinations_connection_ratio=0.15, seed=0):
    """

    Parameters
    ----------
    number_of_nodes : number of nodes (potential origins)
    origins_to_nodes_ratio : float, indicates what fraction of nodes are assumed to be origins
    seed : seed for numpy random
    origins_to_destinations_connection_ratio :

    Returns
    -------
    od_matrix

    """
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    od_connections = int(origins_to_destinations_connection_ratio)
    number_of_origins = int(origins_to_nodes_ratio * number_of_nodes)
    origins = np.random.choice(np.arange(number_of_nodes), size=number_of_origins, replace=False)
    destinations = np.random.choice(np.arange(number_of_nodes), size=number_of_origins, replace=False)
    for origin in origins:
        # randomly sample how many and which destinations this origin is connected to
        number_of_destinations = int(np.random.gumbel(loc=origins_to_destinations_connection_ratio,
                                                      scale=origins_to_destinations_connection_ratio / 2) * len(
            destinations))
        if number_of_destinations < 0: continue
        if number_of_destinations > len(destinations): number_of_destinations = len(destinations)
        destinations_by_origin = np.random.choice(destinations, size=number_of_destinations, replace=False)
        rand_od.rows[origin] = list(destinations_by_origin)
        rand_od.data[origin] = [int(np.random.random() * 2000) for _ in destinations_by_origin]
    return rand_od


def generate_od_fixed(number_of_nodes, number_of_od_values, seed=0):
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
    np.random.seed(seed)
    arr = np.arange(number_of_nodes * number_of_nodes)
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
    rand_od = lil_matrix((number_of_nodes, number_of_nodes), dtype=np.uint32)
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
