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
import datetime
import os
import warnings
from inspect import signature
from itertools import chain
from warnings import warn

import networkx as nx
import numpy as np
import osmnx as ox
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import Spacer, column, row
from bokeh.models import Circle, HoverTool, OpenURL, TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import MultiPolygons, Patches
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import ColumnDataSource, figure
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon

from dyntapy.settings import parameters
from dyntapy.utilities import __create_green_to_red_cm

traffic_cm = __create_green_to_red_cm()

link_highlight_colors = parameters.visualization.link_highlight_colors
node_highlight_color = parameters.visualization.node_highlight_color
node_color = parameters.visualization.node_color
centroid_color = parameters.visualization.centroid_color
node_size = parameters.visualization.node_size

tile_provider = "CARTODBPOSITRON"  # tile


def _get_output_file(plot_name: str):
    dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return os.getcwd() + os.path.sep + f"{plot_name}_{dt_string}.html"


def _process_plot_arguments(
    g, title, notebook, euclidean, link_kwargs, node_kwargs, max_edge_width
):
    # process all arguments that are shared between dynamic and static
    for _, _, data in g.edges.data():
        if "x_coord" in data:
            data["x"] = data["x_coord"]
            data["y"] = data["y_coord"]
    for _, data in g.nodes.data():
        if "x_coord" in data:
            data["x"] = data["x_coord"]
            data["y"] = data["y_coord"]
    needs_relabelling = False
    try:
        linkids = sorted([val for _, _, val in g.edges.data("link_id")])
        if (
            None in linkids
            or linkids[0] != 0
            or not all(i == j - 1 for i, j in zip(linkids, linkids[1:]))
        ):
            needs_relabelling = True
    except TypeError:
        raise ValueError("Links in g were not labelled, use relabel_graph function")
    if needs_relabelling:
        raise ValueError(
            "Links in g were not labelled correctly, use relabel_graph function"
        )
        # they have got to be
        # starting at 0 & consecutively labelled integers
    if not euclidean:
        plot = figure(
            height=parameters.visualization.plot_size,
            width=parameters.visualization.plot_size,
            x_axis_type="mercator",
            y_axis_type="mercator",
            aspect_ratio=1,
            toolbar_location="below",
        )
        plot.add_tile(tile_provider, retina=True)
        tmp = nx.MultiDiGraph(g)
        tmp = ox.project_graph(
            tmp, CRS.from_user_input(3857)
        )  # from lan lot to web mercator

        link_width_scaling = parameters.visualization.link_width_scaling
    else:

        link_width_scaling = parameters.visualization.link_width_scaling_euclidean
        plot = figure(
            height=parameters.visualization.plot_size,
            width=parameters.visualization.plot_size,
            aspect_ratio=1,
            toolbar_location="below",
        )
        tmp = g

    link_width_scaling = max_edge_width * link_width_scaling
    if title is None:
        title = "network_plot"
    plot.title.text = title
    if notebook:
        output_notebook(hide_banner=True)
    else:
        output_file(_get_output_file(title))
    # process link and node kwargs to right formatting in the case of
    # multidimensional arrays, bokeh can only parse list of lists
    for key, item in zip(link_kwargs.keys(), link_kwargs.values()):
        if type(link_kwargs[key]) == np.ndarray:
            if np.issubdtype(item.dtype, np.floating):
                link_kwargs[key] = (
                    item.astype(np.float64)
                    .round(parameters.visualization.rounding_digits)
                    .tolist()
                )
            else:
                link_kwargs[key] = item.tolist()
    for key, item in zip(node_kwargs.keys(), node_kwargs.values()):
        if type(node_kwargs[key]) == np.ndarray:
            if np.issubdtype(item.dtype, np.floating):
                node_kwargs[key] = (
                    item.astype(np.float64)
                    .round(parameters.visualization.rounding_digits)
                    .tolist()
                )
            else:
                node_kwargs[key] = item.tolist()
    return plot, tmp, link_width_scaling


def show_network(
    g,
    flows=None,
    link_kwargs=dict(),
    node_kwargs=dict(),
    highlight_links=np.array([]),
    highlight_nodes=np.array([]),
    euclidean=False,
    toy_network=False,
    title=None,
    notebook=False,
    show_nodes=True,
    return_plot=False,
    max_edge_width=1,
):
    """
    Visualizing a network with static attributes in a .html.

    Parameters
    ----------
    g: networkx.DiGraph
    flows: numpy.ndarray, optional
        float, 1D - flows to be visualized
    link_kwargs: dict, optional
        additional link information to be displayed
        key: str, value: numpy.ndarray - additional link information to be
        displayed
    node_kwargs: dict, optional
        key: str, value: numpy.ndarray - additional node information to be
        displayed
    highlight_links: numpy.ndarray or list, optional
        int, 1D or 2D - links to highlight
    highlight_nodes: numpy.ndarray or list, optional
        int, 1D or 2D - nodes to highlight
    euclidean: bool, optional
        set to True if coordinates in graph are euclidean.
    toy_network: bool, optional
        deprecated, use euclidean instead.
    title: str, optional
    notebook: bool, optional
        set to True if the plot should be rendered in a notebook.
    show_nodes: bool, optional
        whether to render nodes
    return_plot: bool, optional
        set to True if the plot object should be returned instead of showing it.
    max_edge_width: float, optional
        defaults to 1, changes the width of the edges by the set factor

    Examples
    --------

    >>> show_network(g, highlight_links=[[2,4],[3,6]])

    will plot the network and highlight links [2,4] in neon pink,
    and [3, 6] in cyan. The order of highlight colors is neon pink, cyan, lime green,
    light blue, orange, gray

    Node highlighting works analogously.

    >>> foo = np.arange(g.number_of_edges())
    >>> bar = np.arange(g.number_of_edges())
    >>> show_network(g, link_kwargs={'foo': foo, 'bar':bar})

    will generate a plot where respective values for `foo` and `bar` can be inspected by
    hovering over the link.
    Note that the string attribute names cannot contain spaces and that the arrays
    must have the correct dimension.

    node_kwargs can be passed on analogously.


    Notes
    -----

    highlight colors can be altered in the settings and have been chosen to still offer
    visibility in a graph with loaded traffic in a green-to-red color map.

    See Also
    -------------

    dyntapy.results.StaticResult

    """

    if toy_network:
        euclidean = toy_network
        warnings.warn(
            "use of toy_network arg is deprecated, use euclidean instead",
            DeprecationWarning,
        )
    plot, tmp, link_width_scaling = _process_plot_arguments(
        g,
        title,
        notebook,
        euclidean,
        link_kwargs,
        node_kwargs,
        max_edge_width=max_edge_width,
    )
    show_flows = True
    if flows is not None:
        pass
    else:
        flows = np.zeros(g.number_of_edges())
        show_flows = False
    max_flow = max(np.max(flows), 1)

    max_width_bokeh, max_width_coords = get_max_edge_width(
        tmp,
        link_width_scaling,
        parameters.visualization.plot_size,
    )

    if type(highlight_links) not in (np.ndarray, list):
        raise ValueError
    c, x, y = _get_colors_and_coords(
        tmp,
        max_width_coords,
        max_flow,
        flows,
        time_step=1,
        highlight_links=highlight_links,
        patch_ratio=parameters.visualization.link_width_min_max_ratio,
    )
    edge_source = _edge_cds(tmp, c, flows, x, y, **link_kwargs)
    edge_renderer = plot.add_glyph(
        edge_source,
        glyph=Patches(
            xs="x",
            ys="y",
            fill_color="color",
            line_color="black",
            line_alpha=0.8,
            fill_alpha=parameters.visualization.link_transparency,
        ),
    )
    edge_tooltips = [
        (item, f"@{item}")
        for item in parameters.visualization.link_keys + list(link_kwargs.keys())
        if item != "flow"
    ]
    if show_flows:
        edge_tooltips.append(("flow", "@flow{(0.00)}"))
    edge_hover = HoverTool(
        show_arrow=False,
        tooltips=edge_tooltips,
        renderers=[edge_renderer],
        description="Link Hover Tool",
    )
    if show_nodes:
        node_source = _node_cds(tmp, highlight_nodes, **node_kwargs)
        node_renderer = plot.add_glyph(
            node_source,
            glyph=Circle(
                x="x",
                y="y",
                size=node_size,
                line_color="black",
                fill_color="color",
                line_alpha=0.4,
                fill_alpha=0.7,
                line_width=node_size / 10,
            ),
        )
        node_tooltips = [
            (item, f"@{item}")
            for item in parameters.visualization.node_keys + list(node_kwargs.keys())
        ]
        node_hover = HoverTool(
            show_arrow=False,
            tooltips=node_tooltips,
            renderers=[node_renderer],
            description="Node Hover Tool",
        )
        plot.add_tools(node_hover)
    plot.add_tools(edge_hover)
    if not return_plot:
        show(plot)
    else:
        return plot


def show_link_od_flows(g: nx.DiGraph, od_flows, **kwargs):
    """
    Visualizing a network with origin destination flows for each link in a .html.

    Parameters
    ----------
    g: networkx.DiGraph
    od_flows: list
        origin destination flows for each link
    kwargs: any
        all the arguments of `show_network` are valid, except for `flows`

    See Also
    --------

    dyntapy.visualization.show_network

    """
    formatted_od_flows = []
    tot_flows = np.zeros(len(od_flows), dtype=np.float64)
    for link in range(len(od_flows)):
        formatted_od_flows.append("")
        for o, d, flow in od_flows[link]:
            tot_flows[link] += flow
            # no line breaks in bokeh due to an upstream issue ..
            # https://discourse.bokeh.org/t/multiline-
            # strings-in-bokeh-datatable-linebreaks-getting-stripped/1078
            formatted_od_flows[link] += f"({o}, {d}, {flow}) "
    if "link_kwargs" in kwargs:
        kwargs["link_kwargs"] = {kwargs["link_kwargs"] | "od_flows": formatted_od_flows}
    else:
        kwargs["link_kwargs"] = {"od_flows": formatted_od_flows}
        show_network(g, flows=tot_flows, **kwargs)


def show_dynamic_network(
    g,
    time,
    flows=None,
    link_kwargs=dict(),
    node_kwargs=dict(),
    toy_network=False,
    euclidean=False,
    highlight_nodes=np.array([]),
    highlight_links=np.array([]),
    title=None,
    notebook=False,
    show_nodes=True,
    return_plot=False,
    max_edge_width=1,
):
    """
    Visualizing a network with dynamic attributes in a .html.

    Parameters
    ----------
    g: networkx.DiGraph
    time: dyntapy.demand.SimulationTime
    flows: numpy.ndarray, optional
        float, 2D - time-dependent flows to be visualized
    link_kwargs: dict, optional
        additional time-dependent link information to be displayed
        key: str, value: np.ndarray - additional time-dependent link information to
        be displayed
    node_kwargs: dict, optional
        key: str, value: np.ndarray - additional time-dependent node information to
        be displayed
    highlight_links: numpy.ndarray, optional
        int, 1D or 2D - links to highlight
    highlight_nodes: numpy.ndarray, optional
        int, 1D or 2D - nodes to highlight
    euclidean: bool, optional
        set to True if coordinates in graph are euclidean.
    toy_network: bool, optional
        deprecated, use euclidean instead.
    title: str, optional
    notebook: bool, optional
        set to True if the plot should be rendered in a notebook.
    show_nodes: bool, optional
        whether to render nodes
    return_plot: bool, optional
        set to True if the plot object should be returned instead of showing it.
    max_edge_width: float, optional
        defaults to 1, changes the width of the edges by the set factor

    Examples
    --------
    highlighting works just as in `show_network` and is not time dependent.

    >>> foo = np.arange((g.number_of_edges(), time.tot_time_steps))
    >>> bar = np.arange((g.number_of_edges(), time.tot_time_steps))
    >>> show_dynamic_network(g, link_kwargs={'foo': foo, 'bar':bar})

    Will generate a plot where respective values for `foo` and `bar` can be inspected by
    hovering over the link.
    The values are updated as the time slider is moved.

    Note that the string attribute names cannot contain spaces and that the arrays
    must have the correct dimension.

    node_kwargs can be passed on analogously.


    Notes
    -----

    highlight colors can be altered in the settings and have been chosen to still offer
    visibility in a graph with loaded traffic in a green-to-red color map.

    See Also
    -------------

    dyntapy.results.DynamicResult

    """

    if toy_network:
        euclidean = toy_network
        warnings.warn(
            "use of toy_network arg is deprecated, use euclidean instead",
            DeprecationWarning,
        )

    plot, tmp, link_width_scaling = _process_plot_arguments(
        g,
        title,
        notebook,
        euclidean,
        link_kwargs,
        node_kwargs,
        max_edge_width=max_edge_width,
    )
    if flows is None:
        if "flows" not in list(link_kwargs.keys()):
            flows = np.zeros((time.tot_time_steps, g.number_of_edges()))
        else:
            flows = link_kwargs["flows"]

    static_link_kwargs = dict()
    static_node_kwargs = dict()
    for key, item in zip(link_kwargs.keys(), link_kwargs.values()):
        if type(item) == list:
            if len(item) == g.number_of_edges:
                static = True
            elif len(item) == time.tot_time_steps and all(
                len(k) == g.number_of_edges() for k in item
            ):
                static = False
            else:
                raise ValueError("dimension mismatch")
            if static:
                static_link_kwargs[key] = link_kwargs[key]
        else:
            raise ValueError("values in link_kwargs need to be converted to list")
    for key in static_link_kwargs.keys():
        del link_kwargs[key]
    for key, item in zip(node_kwargs.keys(), node_kwargs.values()):
        if type(node_kwargs[key]) == np.ndarray:
            if item.shape == (g.number_of_nodes(),):
                static = True
            elif (
                item.shape[0] == time.tot_time_steps
                and item.shape[1] == g.number_of_nodes()
            ):
                static = False
            else:
                raise ValueError("dimension mismatch")
            if static:
                static_node_kwargs[key] = node_kwargs[key]
        else:
            raise ValueError("values in node_kwargs need to be numpy.ndarray")
    for key in static_node_kwargs.keys():
        del node_kwargs[key]

    # adding different coordinate attribute names to comply with osmnx

    max_flow = min(np.max(flows), 8000)  # weeding out numerical errors
    max_width_bokeh, max_width_coords = get_max_edge_width(
        tmp,
        link_width_scaling,
        parameters.visualization.plot_size,
    )
    # calculate all colors and coordinates for the different time dependent flows
    all_colors = []
    all_x = []
    all_y = []
    for t in range(time.tot_time_steps):
        c, x, y = _get_colors_and_coords(
            tmp, max_width_coords, max_flow, flows[t], time.step_size, highlight_links
        )
        all_x.append(x)
        all_y.append(y)
        all_colors.append(c)
    link_kwargs_t0 = {
        key: val[0] for key, val in zip(link_kwargs.keys(), link_kwargs.values())
    }  # getting time step zero for all
    link_kwargs_t0 = {**link_kwargs_t0, **static_link_kwargs}
    edge_source = _edge_cds(
        tmp,
        all_colors[0],
        flows[0],
        all_x[0],
        all_y[0],
        step_size=time.step_size,
        **link_kwargs_t0,
    )
    node_kwargs_t0 = {
        key: val[0] for key, val in zip(node_kwargs.keys(), node_kwargs.values())
    }
    node_kwargs_t0 = {**node_kwargs_t0, **static_node_kwargs}
    node_source = _node_cds(tmp, highlight_nodes, **node_kwargs_t0)

    edge_renderer = plot.add_glyph(
        edge_source,
        glyph=Patches(
            xs="x",
            ys="y",
            fill_alpha=parameters.visualization.link_transparency,
            fill_color="color",
            line_color="black",
            line_alpha=0.4,
            line_width=0.4,
        ),
    )
    edge_tooltips = [
        (item, f"@{item}")
        for item in parameters.visualization.link_keys
        + list(link_kwargs.keys())
        + list(static_link_kwargs.keys())
        if item != "flow"
    ]
    # link_kwargs_tooltips = [(item, '@' + str(item) + '{(0.00)}') for item in list(
    # link_kwargs.keys())]
    # edge_tooltips = edge_tooltips + link_kwargs_tooltips
    edge_tooltips.append(("flow", "@flow{(0.00)}"))
    if show_nodes:
        node_renderer = plot.add_glyph(
            node_source,
            glyph=Circle(
                x="x",
                y="y",
                size=node_size,
                fill_color="color",
                line_alpha=0.4,
                fill_alpha=0.7,
                line_color="black",
                line_width=node_size / 10,
            ),
        )
        node_tooltips = [
            (item, f"@{item}")
            for item in parameters.visualization.node_keys
            + list(node_kwargs.keys())
            + list(static_node_kwargs.keys())
        ]
        node_hover = HoverTool(
            show_arrow=False,
            tooltips=node_tooltips,
            renderers=[node_renderer],
            description="Node Hover Tool",
        )
        plot.add_tools(node_hover)
    # node_kwargs_tooltips = [(item, '@' + str(item) + '{(0.00)}') for item in list(
    # node_kwargs.keys())]
    # node_tooltips= node_tooltips+node_kwargs_tooltips

    edge_hover = HoverTool(
        show_arrow=False,
        tooltips=edge_tooltips,
        renderers=[edge_renderer],
        description="Link Hover Tool",
    )

    text_input = TextInput(title="Add new graph title", value="")
    text_input.js_link("value", plot.title, "text")
    time_slider = Slider(
        start=0,
        end=time.end - time.step_size,
        value=0,
        step=time.step_size,
        title="time",
    )

    plot.add_tools(edge_hover)

    # Set up callbacks
    # all arguments need to be lists
    link_call_back = CustomJS(
        args=dict(
            source=edge_source,
            all_x=all_x,
            all_y=all_y,
            flows=flows.tolist(),
            all_colors=all_colors,
            link_kwargs=link_kwargs,
            step_size=time.step_size,
        ),
        code="""
        var data = source.data;
        var t = Math.round(cb_obj.value/step_size)
        for(var key in link_kwargs) {
            var value = link_kwargs[key][t];
            data[key] = value
            }

        data['x'] = all_x[t]
        data['y'] = all_y[t]
        data['color'] = all_colors[t]
        data['flow']  = flows[t]
        source.change.emit();
    """,
    )

    # all arguments need to be lists
    node_call_back = CustomJS(
        args=dict(
            source=node_source, node_kwargs=node_kwargs, step_size=time.step_size
        ),
        code="""
            var data = source.data;
            var t = cb_obj.value/step_size
            for(var key in node_kwargs) {
                var value = node_kwargs[key][t];
                data[key] = value
                }

            source.change.emit();
        """,
    )
    time_slider.js_on_change("value", link_call_back)
    if show_nodes:
        time_slider.js_on_change("value", node_call_back)
    layout = row(plot, column(text_input, Spacer(height=40), time_slider))
    if return_plot:
        return layout
    else:
        show(layout)


def get_max_edge_width(g, scaling, plot_size):
    node_x = [x for _, x in g.nodes.data("x")]
    node_y = [y for _, y in g.nodes.data("y")]
    min_x, max_x, min_y, max_y = min(node_x), max(node_x), min(node_y), max(node_y)
    max_width_coords = scaling * (0.5 * (max_x - min_x) + 0.5 * (max_y - min_y))
    max_width_bokeh = plot_size * scaling
    return max_width_bokeh, max_width_coords


def show_demand(
    g,
    title=None,
    notebook=False,
    euclidean=False,
    toy_network=False,
    highlight_nodes=[],
    return_plot=False,
    max_edge_width=1,
):
    """

    visualize demand on a map

    Parameters
    ----------
    g: networkx.DiGraph
    title: str
    notebook: bool, optional
        set to True if the plot should be rendered in a notebook.
    euclidean: bool, optional
        set to True, if 'x_coord' and 'y_coord' in g are euclidean.
    toy_network: bool, optional
        deprecated, use euclidean instead
    highlight_nodes: numpy.ndarray or list, optional
        int, 1D - nodes to highlight
    return_plot: bool, optional
        set to True if the plot object should be returned instead of showing it.
    max_edge_width: float, optional
        defaults to 1, changes the width of the edges by the set factor

    Examples
    --------

    Given an `od_matrix` and coordinates in longitude `X` and latitude `Y` it is
    straightforward to visualize the travel pattern.

    >>> od_graph_from_matrix(od_matrix, X, Y)
    >>> show_demand(g)

    See Also
    --------

    dyntapy.demand_data.od_graph_from_matrix

    """
    if toy_network:
        euclidean = toy_network
        warnings.warn(
            "use of toy_network arg is deprecated, use euclidean instead",
            DeprecationWarning,
        )

    for _, _, data in g.edges.data():
        if "x_coord" in data:
            data["x"] = data["x_coord"]
            data["y"] = data["y_coord"]
    for _, data in g.nodes.data():
        if "x_coord" in data:
            data["x"] = data["x_coord"]
            data["y"] = data["y_coord"]

    if title is None:
        title = "demand_plot"
    if notebook:
        output_notebook(hide_banner=True)
    else:
        output_file(filename=_get_output_file(title))
    if not euclidean:
        g.graph["crs"] = "epsg:4326"
        tmp = nx.MultiDiGraph(g)
        tmp = ox.project_graph(tmp, CRS.from_user_input(3857))
        plot = figure(
            inner_height=parameters.visualization.plot_size,
            inner_width=parameters.visualization.plot_size,
            x_axis_type="mercator",
            y_axis_type="mercator",
            aspect_ratio=1,
            toolbar_location="below",
        )
        plot.add_tile(tile_provider, retina=True)
        link_width_scaling = (
            max_edge_width * parameters.visualization.link_width_scaling
        )
    else:
        tmp = (
            g  # projection not needed for toy networks, coordinates are plain
            # cartesian
        )
        plot = figure(
            # inner_height=parameters.visualization.plot_size,
            # inner_width=parameters.visualization.plot_size,
            aspect_ratio=1,
            toolbar_location="below",
        )
        link_width_scaling = parameters.visualization.link_width_scaling_euclidean
    plot.title.text = title
    max_width_bokeh, max_width_coords = get_max_edge_width(
        tmp,
        link_width_scaling,
        parameters.visualization.plot_size,
    )
    min_width_coords = max_width_coords / 10
    all_flow = [flow for u, v, flow in tmp.edges.data("flow") if u != v]
    max_flow = max(all_flow)
    x_list, y_list = [], []
    for u, v, data in tmp.edges.data():
        if u != v:  # not showing intrazonal traffic
            flow = data["flow"]
            width_coords = min_width_coords + (max_width_coords - min_width_coords) * (
                flow / max_flow
            )
            x, y = __build_patch_buffered(
                tmp.nodes[u]["x"],
                tmp.nodes[u]["y"],
                tmp.nodes[v]["x"],
                tmp.nodes[v]["y"],
                data,
                width_coords,
            )
            x_list.append(x)
            y_list.append(y)

    nodes_to_highlight = np.full(tmp.number_of_nodes(), False)
    nodes_to_highlight[highlight_nodes] = True
    node_colors = [
        node_highlight_color if nodes_to_highlight[idx] else node_color
        for idx in list(tmp.nodes.keys())
    ]

    node_source = ColumnDataSource(
        data=dict(
            color=node_colors,
            x=[x for _, x in tmp.nodes.data("x")],
            y=[y for _, y in tmp.nodes.data("y")],
            centroid_id=list(tmp.nodes.keys()),
        )
    )
    edge_source = ColumnDataSource(data=dict(flow=all_flow, x=x_list, y=y_list))
    # text_input = TextInput(title="Add new graph title", value='')
    # text_input.js_link('value', plot.title, 'text')
    edge_renderer = plot.add_glyph(
        edge_source,
        glyph=Patches(
            xs="x", ys="y", fill_color="green", line_color="black", line_alpha=0.8
        ),
    )
    edge_tooltips = [("flow", "@flow{(0.0)}")]
    node_renderer = plot.add_glyph(
        node_source,
        glyph=Circle(
            x="x",
            y="y",
            size=node_size * 2,
            fill_color="color",
            line_color="black",
            line_alpha=0.4,
            fill_alpha=0.7,
            line_width=node_size / 10,
        ),
    )
    node_tooltips = [(item, f"@{item}") for item in ["x", "y", "centroid_id"]]
    edge_hover = HoverTool(
        show_arrow=False,
        tooltips=edge_tooltips,
        renderers=[edge_renderer],
        description="Link Hover Tool",
    )
    node_hover = HoverTool(
        show_arrow=False,
        tooltips=node_tooltips,
        renderers=[node_renderer],
        description="Node Hover Tool",
    )
    plot.add_tools(node_hover, edge_hover)
    text_input = TextInput(title="Add new graph title", value="")
    text_input.js_link("value", plot.title, "text")
    layout = row(plot, text_input)
    if return_plot:
        return plot
    else:
        show(layout)


def _node_cds(g, highlight_nodes=np.array([]), **kwargs):
    visualization_keys = parameters.visualization.node_keys
    node_dict = dict()
    node_colors = [node_color for _ in range(g.number_of_nodes())]
    for _, data in sorted(g.nodes(data=True), key=lambda t: t[1]["node_id"]):
        if data.get("centroid", False):
            node_colors[data["node_id"]] = centroid_color
    for node in highlight_nodes:
        node_colors[node] = node_highlight_color
    node_dict["color"] = node_colors
    for attr_key in visualization_keys + ["x", "y"]:
        values = [
            node_attr[attr_key] if attr_key in node_attr.keys() else "None"
            for _, node_attr in sorted(
                g.nodes(data=True), key=lambda t: t[1]["node_id"]
            )
        ]
        node_dict[attr_key] = values
    node_dict = {**node_dict, **kwargs}
    return ColumnDataSource(data=node_dict)


def _edge_cds(g, color, flow, x, y, step_size=1.0, **kwargs):
    visualization_keys = parameters.visualization.link_keys
    edge_dict = dict()
    for attr_key in visualization_keys:
        values = [
            edge_attr[attr_key] if attr_key in edge_attr.keys() else "None"
            for _, _, edge_attr in sorted(
                g.edges(data=True), key=lambda t: t[2]["link_id"]
            )
        ]
        edge_dict[attr_key] = values
    edge_dict["capacity"] = (np.array(edge_dict["capacity"]) * step_size).tolist()
    edge_dict["color"] = color
    edge_dict["flow"] = flow
    edge_dict["x"] = x
    edge_dict["y"] = y
    edge_dict = {**edge_dict, **kwargs}
    return ColumnDataSource(data=edge_dict)


def _get_colors_and_coords(
    g,
    max_width_coords,
    max_flow,
    flows,
    time_step,
    highlight_links: object = np.array([]),
    patch_ratio=parameters.visualization.link_width_min_max_ratio,
):
    nr_of_colors = len(traffic_cm)
    min_width_coords = max_width_coords * patch_ratio
    if (
        max_flow == 0
    ):  # geometries cannot be computed, may sometimes happen in debugging.
        max_flow = 1
    colors = []
    x_list = []
    y_list = []

    for u, v, data in sorted(g.edges(data=True), key=lambda t: t[2]["link_id"]):
        try:
            flow = flows[data["link_id"]]
            color = traffic_cm[
                int(
                    np.ceil(
                        np.abs(flows[data["link_id"]])
                        / (data["capacity"] * time_step)
                        * nr_of_colors
                    )
                )
            ]
        except IndexError:
            color = traffic_cm[-1]  # flow larger than capacity!
            flow = data["capacity"]
        except KeyError:  # capacity or flow not defined
            color = traffic_cm[0]
            flow = 0
        colors.append(color)
        try:
            if flow > 0:
                loaded = 1
                width_coords = (
                    min_width_coords
                    + min_width_coords * loaded
                    + (max_width_coords - 2 * min_width_coords)
                    * (np.abs(flows[data["link_id"]]) / max_flow)
                )
            else:
                width_coords = min_width_coords
            # width_bokeh = min_width_bokeh + (max_width_bokeh - min_width_bokeh) * (
            # data['flow'] / max_flow)
        except KeyError or UnboundLocalError or IndexError:  # flow not defined..,
            # no width scaling possible
            width_coords = min_width_coords
            # width_bokeh = min_width_bokeh
        # edge_dict['width'].append(width_bokeh)

        x1, y1, x2, y2 = (
            g.nodes[u]["x"],
            g.nodes[u]["y"],
            g.nodes[v]["x"],
            g.nodes[v]["y"],
        )
        x, y = __build_patch_buffered(x1, y1, x2, y2, data, width_coords)
        x_list.append(x)
        y_list.append(y)

    if type(highlight_links) == np.ndarray or (
        type(highlight_links) == list
        and all(
            isinstance(x, np.integer) or isinstance(x, int) for x in highlight_links
        )
    ):
        # single list or array containing integers
        for link in highlight_links:
            colors[link] = link_highlight_colors[0]
    elif type(highlight_links) == list:
        # list of lists, list of arrays for multiple colors
        if not all(
            isinstance(x, np.ndarray) or isinstance(x, list) for x in highlight_links
        ):
            raise TypeError
        elif len(highlight_links) > len(link_highlight_colors):
            raise ValueError(
                f"only {len(link_highlight_colors)} different colors are supported."
            )
        else:
            for links, color in zip(highlight_links, link_highlight_colors):
                for link in links:
                    colors[link] = color

    return colors, x_list, y_list


def __build_patch_buffered(x1, y1, x2, y2, data, width_coords):
    # buffers around the line in one direction.
    # returns a list of x and y coordinates that form the boundary of the generated
    # polygon.

    if "geometry" in data:
        ls = data["geometry"]
    else:
        ls = LineString([[x1, y1], [x2, y2]])
    poly = ls.buffer(-width_coords, single_sided=True)
    if isinstance(poly, MultiPolygon) or len(poly.interiors) > 0:  # Lines with very
        # steep curves or self-intersections yield either multiple polygons or
        # polygons with holes, in either case we take the straight line instead.
        # TODO: investigate support for MultiPolygons with holes in Bokeh
        ls = LineString([[x1, y1], [x2, y2]])
        poly = ls.buffer(-width_coords, single_sided=True)
    return poly.exterior.xy[0].tolist(), poly.exterior.xy[1].tolist()
