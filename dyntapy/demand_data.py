#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2021 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
import itertools
from collections import deque
from json import loads

import geojson
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from warnings import warn
from numba import njit, prange
from geojson import Feature, FeatureCollection, dumps
from osmnx.distance import euclidean_dist_vec, great_circle_vec
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from dyntapy.settings import parameters
from dyntapy.supply_data import default_buffer_dist
from dyntapy.utilities import log


def generate_random_od_graph(tot_ods, name, g, max_flow=2000, seed=0):
    """
    generates a random od-graph for a place

    Parameters
    ----------
    tot_ods : int
        total number of OD pairs to be generated
    name : str
        name of the city or region to geocode and sample from
    g : networkx.DiGraph
    max_flow : float, optional
        maximum demand for any OD pair
    seed : int, optional
        random seed

    Returns
    -------

    od_graph: networkx.DiGraph
        graph with centroids as nodes with specified coordinates as 'x_coord' and
        'y_coord'. For each OD pair with a non-zero demand there
        is a link with a corresponding 'flow' element as read from the OD matrix.
    """
    json = generate_od_xy(tot_ods, name, max_flow, seed)
    return parse_demand(json, g)


def generate_od_xy(tot_ods, name: str, max_flow=2000, seed=0):
    """
    generate random demand for a place in geojson format

    Parameters
    ----------
    tot_ods : int
        total number of OD pairs to be generated
    name : str
        name of the city or region to geocode and sample from
    max_flow : float, optional
        maximum demand for any OD pair
    seed : int, optional
        random seed
    Returns
    -------
    geojson
        containing LineStrings with a 'flow' attribute
    """
    # 4326 : WGS84
    # 3857 : web mercator
    np.random.seed(seed)
    my_gdf: gpd.geodataframe.GeoDataFrame = ox.geocode_to_gdf(name)
    tot_points = (
        20 * tot_ods
    )  # could be improved by using the area ratio between bbox and polygon for scaling
    X = (
        np.random.random(tot_points) * (my_gdf.bbox_east[0] - my_gdf.bbox_west[0])
        + my_gdf.bbox_west[0]
    )
    # np.normal.normal means uniform distribution between 0 and 1
    # , can easily be replaced (gumbel, gaussian..)
    Y = (
        np.random.random(tot_points) * (my_gdf.bbox_north[0] - my_gdf.bbox_south[0])
        + my_gdf.bbox_south[0]
    )
    points = [Point(x, y) for x, y in zip(X, Y)]
    my_points = gpd.geoseries.GeoSeries(points, crs=4326)
    valid_points = my_points[
        my_points.within(my_gdf.loc[0, "geometry"])
    ]  # bounding box typically doesn't align
    # with polygon extend so we ought to check which points are inside
    X = np.array(valid_points.geometry.x[: tot_ods * 2])
    Y = np.array(valid_points.geometry.y[: tot_ods * 2])
    destinations = [(x, y) for x, y in zip(X[:tot_ods], Y[:tot_ods])]
    origins = [(x, y) for x, y in zip(X[tot_ods:], Y[tot_ods:])]
    vals = np.random.random(tot_ods) * max_flow
    line_strings = [
        LineString([[origin[0], origin[1]], [destination[0], destination[1]]])
        for origin, destination in zip(origins, destinations)
    ]
    tmp = [{"flow": f} for f in vals]
    my_features = [
        Feature(geometry=my_linestring, properties=my_tmp)
        for my_linestring, my_tmp in zip(line_strings, tmp)
    ]
    fc = FeatureCollection(my_features)
    return dumps(fc)


def __set_connector_default_attr(g):
    connectors = [(u, v) for u, v, data in g.edges.data() if "connector" in data]
    connector_sg = nx.edge_subgraph(g, connectors)
    for edge in connector_sg.edges:
        u, v = edge
        g[u][v]["maxspeed"] = default_connector_speed
        g[u][v]["capacity"] = default_connector_capacity
        g[u][v]["lanes"] = default_connector_lanes


def _check_centroid_connectivity(g: nx.DiGraph):
    """
    verifies if each centroid has at least one connector
    Parameters
    ----------
    g : networkx.Digraph

    """
    centroids = [u for u, data_dict in g.nodes(data=True) if "centroid" in data_dict]
    tot_out_c = [__count_iter_items(g.successors(n)) for n in centroids]
    tot_in_c = [__count_iter_items(g.predecessors(n)) for n in centroids]
    if min(tot_out_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_out_c) if val == 0]
        raise ValueError(
            f"these centroids do not have an "
            f"outgoing connector: {disconnected_centroids}"
        )
    if min(tot_in_c) == 0:
        disconnected_centroids = [i for i, val in enumerate(tot_in_c) if val == 0]
        raise ValueError(
            f"these centroids do not have an incoming "
            f"connector: {disconnected_centroids}"
        )


def od_graph_from_matrix(od_matrix: np.ndarray, X, Y):
    """
    creates od_graph from od_matrix and centroid locations

    Parameters
    ----------

    od_matrix : numpy.ndarray
        float, 2D
    X : numpy.ndarray
        float, 1D - lon of centroid locations
    Y : numpy.ndarray
        float, 1D - lat of centroid locations

    Returns
    -------
    od_graph: networkx.DiGraph
        graph with centroids as nodes with specified coordinates as 'x_coord' and
        'y_coord'. For each OD pair with a non-zero demand there
        is a link with a corresponding 'flow' element as read from the OD matrix.


    """
    if od_matrix.shape != (len(X), len(X)):
        raise ValueError("dimensions of centroid locations and OD matrix incompatible")
    g = nx.DiGraph()
    nodes = [(u, {"x_coord": p[0], "y_coord": p[1]}) for u, p in enumerate(zip(X, Y))]
    g.add_nodes_from(nodes)
    edges = [
        (u, v, {"flow": od_matrix[u, v]})
        for u, v in np.argwhere(od_matrix > 0)
        if u != v
    ]
    g.add_edges_from(edges)
    return g


default_centroid_spacing = parameters.demand.default_centroid_spacing


def get_centroid_grid_coords(name: str, spacing=default_centroid_spacing):
    """

    creates centroids on a grid that overlap with the polygon
    that is associated with city or region specified


    Parameters
    ----------
    name : str,
        name of the city to be used as reference polygon
    spacing : float, optional
        distance between two adjacent centroids on the grid

    Returns
    -------
    X: numpy.ndarray
        float, 1D
        lon of centroid locations
    Y: numpy.ndarray
        float, 1D
        lat of centroid locations
    """
    my_gdf = ox.geocode_to_gdf(name)
    range_ns_meters = great_circle_vec(
        my_gdf.bbox_north[0],
        my_gdf.bbox_east[0],
        my_gdf.bbox_south[0],
        my_gdf.bbox_east[0],
    )
    range_ew_meters = great_circle_vec(
        my_gdf.bbox_east[0],
        my_gdf.bbox_south[0],
        my_gdf.bbox_west[0],
        my_gdf.bbox_south[0],
    )
    ns_tics = np.linspace(
        my_gdf.bbox_south[0],
        my_gdf.bbox_north[0],
        int(np.floor(range_ns_meters / spacing)),
    )
    ew_tics = np.linspace(
        my_gdf.bbox_west[0],
        my_gdf.bbox_east[0],
        int(np.floor(range_ew_meters / spacing)),
    )
    grid = np.meshgrid(ew_tics, ns_tics)
    X = grid[0].flatten()
    Y = grid[1].flatten()
    points = [Point(x, y) for x, y in zip(X, Y)]
    points = gpd.geoseries.GeoSeries(points, crs=4326)
    centroids = points[points.within(my_gdf.loc[0, "geometry"])]
    x, y = np.array(centroids.x), np.array(centroids.y)
    log(f"found {x.size} centroids at {spacing=} meter", to_console=True)
    return x, y


def add_centroids(g, X, Y, k=1, method="turn", euclidean=False, **kwargs):
    """

    Adds centroids to g.

    Parameters
    ----------
    g : networkx.Digraph
        road network graph, containing only road network edges and nodes
    X : numpy.ndarray
        float, 1D - lon of centroids
    Y : numpy.ndarray
        float, 1D - lat of centroids
    k : int
        number of road network nodes to connect to per centroid.
    method : {'turn' , 'link'}
        whether to add link or turn connectors
    euclidean : bool, optional
        set to True for toy networks that use the euclidean coordinate system
    **kwargs : iterable, optional
        any keyword arguments are passed as additional attributes into the graph and
        appear as attributes of the centroids.
        They are assumed to be iterable and of
        the same length as X.

    Returns
    -------
    networkx.DiGraph
        new graph with centroids and connectors

    Notes
    -----
    if method is 'link' k*2 connectors are added per centroid, one for each direction.
    if method is 'turn'k*2+2 connectors are added per centroid. There is another
    artificial node between the centroid and the first intersection node. All
    connector turns share the first starting link from centroid to this artificial node.

    The route choice algorithms for DTA within dyntapy rely on iterative computations on
    the link-turn graph rather than the node-link graph.
    Within the network one can take the current link and evaluate the options as the
    set of all outgoing turns.
    Evaluating the choice of the next turn to take from a centroid node without a
    dummy starting link and turns is cumbersome because the structure differs.
    We add these dummy turns to keep the algorithms simpler.



    See Also
    --------

    dyntapy.demand_data.add_connectors

    """
    if len(kwargs) != 0:
        for key, val in zip(kwargs.keys(), kwargs.values()):
            try:
                iter(val)
            except TypeError:
                raise TypeError(f"{key} is not iterable")
            if len(val) != len(X):
                raise ValueError(f"{key} has the wrong dimension")
    new_g = nx.DiGraph()
    new_g.graph = g.graph.copy()
    last_intersection_node = max(g.nodes)
    attributes = {"x_coord": X, "y_coord": Y, **kwargs}
    new_centroids = [
        (
            u + last_intersection_node + 1,
            {
                "centroid": True,
                **{
                    attr: itm[u]
                    for attr, itm in zip(attributes.keys(), attributes.values())
                },
            },
        )
        for u in range(X.size)
    ]
    for u, data in new_centroids:  # first centroids, then intersection nodes for order
        new_g.add_node(u, **data)
    for u, data in g.nodes.data():
        new_g.add_node(u, **data)
    for u, v, data in g.edges.data():
        new_g.add_edge(u, v, **data)
    if method not in ["link", "turn"]:
        raise ValueError
    if method == "link":
        for u, data in new_centroids:
            add_connectors(data["x_coord"], data["y_coord"], u, k, g, new_g, euclidean)

    if method == "turn":
        j0 = max(new_g.nodes) + 1
        for j, (u, data) in enumerate(new_centroids):
            if euclidean:
                delta_y = 0.1  # assuming that toy networks work with unit lengths
            else:
                delta_y = (
                    (100 / 6378137) * 180 / np.pi
                )  # placing the artificial connecting node approx. 100 meters north
            new_g.add_node(
                j0 + j,
                **{"x_coord": data["x_coord"], "y_coord": data["y_coord"] + delta_y},
            )
            new_g.add_edge(
                u,
                j0 + j,
                **{
                    "connector": True,
                    "length": delta_y,
                    "free_speed": default_connector_speed,
                    "lanes": default_connector_lanes,
                    "capacity": default_connector_capacity,
                    "link_type": np.int8(1),
                },
            )
            new_g.add_edge(
                j0 + j,
                u,
                **{
                    "connector": True,
                    "length": delta_y,
                    "free_speed": default_connector_speed,
                    "lanes": default_connector_lanes,
                    "capacity": default_connector_capacity,
                    "link_type": np.int8(-1),
                },
            )
            add_connectors(
                data["x_coord"],
                data["y_coord"],
                j0 + j,
                k,
                g,
                new_g,
                euclidean,
            )

    return new_g


def auto_configured_centroids(
    place,
    buffer_dist_close,
    buffer_dist_extended,
    inner_city_centroid_spacing=default_centroid_spacing,
):
    """
    generates centroids for the inner and extended study area from OpenStreetMap.
    The inner area is filled with a grid.
    The outer buffers are queried for settlements via OSMs 'place' tag.

    Parameters
    ----------
    place: str
        name of the city or region to buffer around.
    buffer_dist_close: float
        width of the inner buffer
    buffer_dist_extended: float
        width of the outer buffer
    inner_city_centroid_spacing: float, optional
        distance between two adjacent centroids on the grid

    Returns
    -------
    X: numpy.ndarray
        float, 1D
        lon of centroid locations
    Y: numpy.ndarray
        float, 1D
        lat of centroid locations
    name: list of strings
        name of the place as specified in OSM
    place: list of strings
        values for OSMs place tag


    Notes
    -----
    The inner buffer is queried for places that are tagged as 'village', 'city',
    or 'town'. The outer buffer is just querying for 'city' or 'town'.

    """
    x_inner, y_inner = get_centroid_grid_coords(
        place, spacing=inner_city_centroid_spacing
    )
    gdf_close = ox.geometries_from_place(
        place, {"place": ["village", "city", "town"]}, buffer_dist=buffer_dist_close
    )
    gdf_extended = ox.geometries_from_place(
        place, {"place": ["city", "town"]}, buffer_dist=buffer_dist_extended
    )
    merged_gdf = pd.concat([gdf_close, gdf_extended])
    G = merged_gdf["geometry"].apply(lambda geom: geom.wkb)
    merged_gdf = merged_gdf.loc[G.drop_duplicates().index]
    merged_gdf = merged_gdf[
        merged_gdf.geometry.type == "Point"
    ]  # considering only points
    names = merged_gdf["name"].tolist()
    x_ext, y_ext = (
        merged_gdf.geometry.apply(lambda x: x.x).to_numpy(dtype=np.float64),
        merged_gdf.geometry.apply(lambda x: x.y).to_numpy(dtype=np.float64),
    )
    place_tags = merged_gdf["place"].tolist()
    place_tags = ["-" for _ in range(len(x_inner))] + place_tags
    names = ["inner_city_centroid" for _ in range(len(x_inner))] + names
    return (
        np.concatenate((x_inner, x_ext)),
        np.concatenate((y_inner, y_ext)),
        names,
        place_tags,
    )


def add_connectors(x, y, u, k, g, new_g, euclidean):
    """
    adding connectors to new_g from starting node u with coordinates x and y to
    nearest k intersection nodes in g.

    Parameters
    ----------
    x : float
        lon
    y : float
        lat
    u : int
        connector's from_node in new_g
    k : int
        number of (bidirectional) connectors to add
    g : networkx.DiGraph
        containing all road network nodes
    new_g : networkx.DiGraph
        containing at least u and all nodes of g
    euclidean : bool
        whether x and y are euclidean

    Notes
    -----

    default attributes of the connectors (speed, capacity, lanes)
    can be changed in dyntapy.settings.

    """
    tmp: nx.MultiDiGraph = g  # calculate distance to road network graph
    og_nodes = list(g.nodes)
    for _ in range(k):
        # find the nearest node j k times, ignoring previously
        # nearest nodes in consequent iterations if
        # multiple connectors are wanted
        if not euclidean:
            v, length = _get_nearest_node(tmp, (y, x), return_dist=True)
        else:
            v, length = _get_nearest_node(
                tmp, (y, x), method="euclidean", return_dist=True
            )
        og_nodes.remove(v)
        tmp = tmp.subgraph(og_nodes)
        source_data = {
            "connector": True,
            "length": length / 1000,
            "free_speed": default_connector_speed,
            "lanes": default_connector_lanes,
            "capacity": default_connector_capacity,
            "link_type": np.int8(1),
        }  # length in km
        sink_data = {
            "connector": True,
            "length": length / 1000,
            "free_speed": default_connector_speed,
            "lanes": default_connector_lanes,
            "capacity": default_connector_capacity,
            "link_type": np.int8(-1),
        }
        new_g.add_edge(u, v, **source_data)
        new_g.add_edge(v, u, **sink_data)


def od_matrix_from_dataframes(
    od_table: pd.DataFrame,
    zoning: gpd.GeoDataFrame,
    origin_column: str,
    destination_column: str,
    zone_column: str,
    flow_column: str,
    return_relabelling=False,
):
    """

    extracts an OD matrix and X and Y coordinates as arrays from a pandas.DataFrame and
    zoning provided as a geopandas.GeoDataFrame.

    It is rather common to receive OD tables in the form of .csv files
    and zoning in the form of .shp files. This function enables one to
    extract the
    full OD matrix after loading these files with pandas and geopandas,
    respectively.

    Parameters
    ----------

    od_table : pandas.DataFrame
        each row should represent one entry in the OD matrix, with `origin_column`,
        `destination_column` and `flow_column` as the relevant columns
    zoning : gpd.GeoDataFrame
        specifies geometries of the zoning, assumed to be in lon lat. The entries in
        `zone_column` should correspond to the entries in `origin_column` and
        `destination_column` in the `od_table`
    origin_column: str
    destination_column: str
    zone_column: str
    flow_column: str
    return_relabelling: bool, optional
        whether to return the mapping between the original zone labels in 'zone_column'
        and the indexes in the returned OD matrix

    Returns
    -------

    od_matrix: numpy.ndarray
        float, 2D
    X: np.ndarray
        float, 1D, lon of centroids
    Y: np.ndarray
        float, 1D, lat of centroids
    mapping: dict, optional

    """

    tot_zones = len(set(zoning[zone_column].to_list()))
    assert len(set(zoning[zone_column].to_list())) == len(zoning[zone_column].to_list())
    # no zone should have two conflicting geometries
    mapping = dict(zip(zoning[zone_column], np.arange(tot_zones)))
    od_table["_origin_idx"] = od_table[origin_column].apply(
        lambda x: mapping.get(x, -1)
    )
    od_table["_destination_idx"] = od_table[destination_column].apply(
        lambda x: mapping.get(x, -1)
    )

    od_matrix = np.zeros((tot_zones, tot_zones), dtype=np.float64)
    try:
        assert tot_zones > len(set(od_table[origin_column].to_list()))
        assert tot_zones > len(set(od_table[destination_column].to_list()))
    except AssertionError:
        warn(
            f"The values found in {zone_column=} do not fully contain all "
            f"values found in "
            f"{origin_column=} and/ or {destination_column=}. There are no "
            f"geometries for some zones! The demand for those zones is omitted"
            f"in the generated OD table."
        )

    flows = od_table[flow_column].to_numpy(dtype=np.float64, copy=True)
    origins = od_table["_origin_idx"].to_numpy()
    destinations = od_table["_destination_idx"].to_numpy()

    @njit
    def parse(_od_matrix, _origins, _destinations, _flows):
        for idx in prange(len(_origins)):
            i = _origins[idx]
            j = _destinations[idx]
            if i != j and i != -1 and j != -1:
                # if either i or j do not have a geometry, we skip them.
                # same for intra zonal traffic
                _od_matrix[i][j] += _flows[idx]

    # extracting centroid geometries from shapefile
    zoning["centroid_x"] = zoning["geometry"].apply(lambda x: x.centroid.x)
    zoning["centroid_y"] = zoning["geometry"].apply(lambda x: x.centroid.y)
    parse(od_matrix, origins, destinations, flows)

    if not return_relabelling:
        return (
            od_matrix,
            zoning["centroid_x"].to_numpy(),
            zoning["centroid_y"].to_numpy(),
        )

    else:
        return (
            od_matrix,
            zoning["centroid_x"].to_numpy(),
            zoning["centroid_y"].to_numpy(),
            mapping,
        )


def parse_demand(data: str, g):
    """
    Maps travel demand to existing closest centroids in g.
    The returned demand pattern is expressed as its own directed graph.

    Parameters
    ----------
    data : geojson
        that contains lineStrings (WGS84) as features,
        each line has an associated 'flow' stored in the properties dictionary
    g : networkx.Digraph


    Returns
    -------

    od_graph: networkx.DiGraph
        graph with centroids as nodes with specified coordinates as 'x_coord' and
        'y_coord'. For each OD pair with a non-zero demand there
        is a link with a corresponding 'flow' element as read from the OD matrix.

    Notes
    -----

    There's no checking on whether the data and `g` correspond to the same geo-coded
    region.

    The corresponding OD table can be retrieved through calling

    >>> networkx.to_scipy_sparse_matrix(od_graph,weight='flow')

    """
    od_graph = nx.MultiDiGraph()
    od_graph.graph["crs"] = "epsg:4326"
    name = g.graph["name"]
    od_graph.graph["name"] = f"mobility flows in {name}"
    od_graph.add_nodes_from(
        [
            (u, data_dict)
            for u, data_dict in g.nodes(data=True)
            if "centroid" in data_dict
        ]
    )
    if od_graph.number_of_nodes() == 0:
        raise ValueError("Graph does not contain any centroids.")
    data = geojson.loads(data)
    gdf = gpd.GeoDataFrame.from_features(data["features"])
    X0 = [gdf.geometry[u].xy[0][0] for u in range(len(gdf))]
    X1 = [gdf.geometry[u].xy[0][1] for u in range(len(gdf))]
    Y0 = [gdf.geometry[u].xy[1][0] for u in range(len(gdf))]
    Y1 = [gdf.geometry[u].xy[1][1] for u in range(len(gdf))]
    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1))
    snapped_centroids, _ = find_nearest_centroids(
        X, Y, od_graph
    )  # snapped centroids are in nx node id space,
    # and not their respective internal centroid ids
    flows = np.array(gdf["flow"])

    ods = snapped_centroids.reshape((int(snapped_centroids.size / 2), 2), order="F")
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

    od_edges = [(od[0], od[1], {"flow": flow}) for od, flow in zip(ods, flows)]
    od_graph.add_edges_from(od_edges)

    return od_graph


def find_nearest_centroids(X, Y, centroid_graph: nx.DiGraph):
    """

    finds the nearest centroids in the graph for a set of locations

    Parameters
    ----------
    X : numpy.ndarray
        longitude of points
    Y : numpy.ndarray
        latitude of points
    centroid_graph : networkx.DiGraph
        with existing centroids,
        coordinates stored as 'x_coord' and 'y_coord' assumed to be lon and lat

    Returns
    -------
    nearest_centroids, numpy.ndarray
        int, 1D
    distances, numpy.ndarray
        in meter

    """
    if centroid_graph.number_of_nodes() == 0:
        raise ValueError("graph does not contain any centroids")
    assert centroid_graph.graph["crs"] == "epsg:4326"
    centroids = pd.DataFrame(
        {
            "x": nx.get_node_attributes(centroid_graph, "x_coord"),
            "y": nx.get_node_attributes(centroid_graph, "y_coord"),
        }
    )
    tree = cKDTree(data=centroids[["x", "y"]], compact_nodes=True, balanced_tree=True)
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
    feature_lists = [loads(my_string)["features"] for my_string in geojsons]
    features = list(itertools.chain(*feature_lists))
    return {"type": "FeatureCollection", "features": features}


def _get_nearest_node(G, point, method="haversine", return_dist=False):
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
        dists = great_circle_vec(
            lat1=df["ref_y"], lng1=df["ref_x"], lat2=df["y"], lng2=df["x"]
        )

    elif method == "euclidean":
        # calculate distances using euclid's formula for projected geometries
        dists = euclidean_dist_vec(
            y1=df["ref_y"], x1=df["ref_x"], y2=df["y"], x2=df["x"]
        )

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


def __count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


default_connector_speed = parameters.demand.default_connector_speed
default_connector_capacity = parameters.demand.default_connector_capacity
default_connector_lanes = parameters.demand.default_connector_lanes


def _places_around_place(
    place, buffer_dist=default_buffer_dist, tags=["city", "town", "village"]
):
    gdf = ox.geometries_from_place(place, {"place": tags}, buffer_dist=buffer_dist)
    names = gdf["name"].tolist()
    x, y = (
        gdf.geometry.apply(lambda x: x.x).to_numpy(dtype=np.float),
        gdf.geometry.apply(lambda x: x.y).to_numpy(dtype=np.float),
    )
    place_tags = gdf["place"].tolist()
    return x, y, names, place_tags
