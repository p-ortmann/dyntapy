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
from inspect import signature
from warnings import warn

import networkx as nx
import numpy as np
import osmnx as ox
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import Spacer, column, row
from bokeh.models import HoverTool, OpenURL, TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.glyphs import Patches
from bokeh.models.markers import Circle
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import ColumnDataSource, figure
from bokeh.tile_providers import Vendors, get_provider
from pyproj import CRS
from shapely.geometry import LineString

from dyntapy.dta.time import SimulationTime
from dyntapy.settings import parameters
from dyntapy.supply_data import relabel_graph
from dyntapy.utilities import __create_green_to_red_cm
from dyntapy.results import StaticResult

traffic_cm = __create_green_to_red_cm()
default_plot_size = parameters.visualization.plot_size
default_notebook_plot_size = parameters.visualization.notebook_plot_size
default_max_links = parameters.visualization.max_links
scaling = parameters.visualization.link_width_scaling
link_highlight_colors = parameters.visualization.link_highlight_colors
node_highlight_color = parameters.visualization.node_highlight_color
node_color = parameters.visualization.node_color
centroid_color = parameters.visualization.centroid_color
node_size = parameters.visualization.node_size
plot_size = default_plot_size


def _get_output_file(plot_name: str):
    dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return os.getcwd() + os.path.sep + f"{plot_name}_{dt_string}.html"


def process_plot_arguments(g, title, notebook, show_nodes, toy_network):
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
            # they have got to be
            # starting at 0 & consecutively labelled integers
    except TypeError:
        needs_relabelling = True
    if needs_relabelling:
        raise ValueError(
            "Links in g were not labelled correctly, use relabel_graph function"
        )
    if not toy_network:
        plot = figure(
            plot_height=plot_size,
            plot_width=plot_size,
            x_axis_type="mercator",
            y_axis_type="mercator",
            aspect_ratio=1,
            toolbar_location="below",
        )
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        plot.add_tile(tile_provider)
        tmp = ox.project_graph(
            g, CRS.from_user_input(3857)
        )  # from lan lot to web mercator
    else:
        plot = figure(
            plot_height=plot_size,
            plot_width=plot_size,
            aspect_ratio=1,
            toolbar_location="below",
        )
        tmp = g

    if title is None:
        title = "network_plot"
    plot.title.text = title
    if notebook:
        output_notebook(hide_banner=True)
    else:
        output_file(_get_output_file(title))
    return plot, tmp


def show_network(
    g: nx.MultiDiGraph,
    result: StaticResult = None,
    flows=None,
    link_kwargs=dict(),
    node_kwargs=dict(),
    highlight_links=np.array([]),
    highlight_nodes=np.array([]),
    toy_network=False,
    title=None,
    notebook=False,
    show_nodes=True,
):
    # adding different coordinate attribute names to use osmnx functions
    plot, tmp = process_plot_arguments(g, title, notebook, show_nodes, toy_network)
    show_flows = True
    if flows is not None:
        pass
    elif result is not None:
        flows = result.flows
    else:
        flows = np.zeros(g.number_of_edges())
        show_flows = False
    max_flow = max(np.max(flows), 1)

    max_width_bokeh, max_width_coords = get_max_edge_width(tmp, scaling, plot_size)

    if type(highlight_links) not in (np.ndarray, list):
        raise ValueError
    c, x, y = _get_colors_and_coords(
        tmp,
        max_width_coords,
        max_flow,
        flows,
        time_step=1,
        highlight_links=highlight_links,
        patch_ratio=3,
    )
    edge_source = _edge_cds(tmp, c, flows, x, y, **link_kwargs)
    edge_renderer = plot.add_glyph(
        edge_source,
        glyph=Patches(
            xs="x", ys="y", fill_color="color", line_color="black", line_alpha=0.8
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
    show(plot)


def show_link_od_flows(g: nx.DiGraph, od_flows, **kwargs):
    # same keyword arguments as show_network
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
    g: nx.DiGraph,
    time: SimulationTime,
    flows=None,
    link_kwargs=dict(),
    node_kwargs=dict(),
    toy_network=False,
    highlight_nodes=np.array([]),
    highlight_links=np.array([]),
    title=None,
    notebook=False,
    show_nodes=True,
):
    """

    Parameters
    ----------
    notebook
    flows
    g : nx.Digraph
    title : str, plot title
    plot_size : height and width measurement in pixel

    Returns
    -------

    """
    plot, tmp = process_plot_arguments(g, title, notebook, show_nodes, toy_network)
    if flows is None:
        if "flows" not in list(link_kwargs.keys()):
            flows = np.zeros((time.tot_time_steps, g.number_of_edges()))
        else:
            flows = link_kwargs["flows"]

    static_link_kwargs = dict()
    static_node_kwargs = dict()
    for key, item in zip(link_kwargs.keys(), link_kwargs.values()):
        if type(link_kwargs[key]) == np.ndarray:
            if item.shape == (g.number_of_edges(),):
                static = True
            elif (
                item.shape[0] == time.tot_time_steps
                and item.shape[1] == g.number_of_edges()
            ):
                static = False
            else:
                raise ValueError("dimension mismatch")

            if np.issubdtype(item.dtype, np.floating):
                link_kwargs[key] = item.astype(np.float64).round(2).tolist()
            else:
                link_kwargs[key] = item.tolist()
            if static:
                static_link_kwargs[key] = link_kwargs[key]
        else:
            raise ValueError("values in link_kwargs need to be numpy.ndarray")
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

            if np.issubdtype(item.dtype, np.floating):
                node_kwargs[key] = item.astype(np.float64).round(2).tolist()
            else:
                node_kwargs[key] = item.tolist()
            if static:
                static_node_kwargs[key] = node_kwargs[key]

        else:
            raise ValueError("values in node_kwargs need to be numpy.ndarray")
    for key in static_node_kwargs.keys():
        del node_kwargs[key]

    # adding different coordinate attribute names to comply with osmnx

    max_flow = min(np.max(flows), 8000)  # weeding out numerical errors
    max_width_bokeh, max_width_coords = get_max_edge_width(tmp, scaling, plot_size)
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
    link_call_back = CustomJS(
        args=dict(
            source=edge_source,
            all_x=all_x,
            all_y=all_y,
            flows=flows,
            all_colors=all_colors,
            link_kwargs=link_kwargs,
            step_size=time.step_size,
        ),
        code="""
        var data = source.data;
        var t = cb_obj.value/step_size
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
    show(layout)


def get_max_edge_width(g, scaling, plot_size):
    node_x = [x for _, x in g.nodes.data("x")]
    node_y = [y for _, y in g.nodes.data("y")]
    min_x, max_x, min_y, max_y = min(node_x), max(node_x), min(node_y), max(node_y)
    max_width_coords = scaling * (0.5 * (max_x - min_x) + 0.5 * (max_y - min_y))
    max_width_bokeh = plot_size * scaling
    return max_width_bokeh, max_width_coords


def show_demand(
    g, title=None, plot_size=default_plot_size, notebook=False, toy_network=False
):
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
        plot_size = 600
    else:
        output_file(filename=_get_output_file(title))
    if not toy_network:
        g.graph["crs"] = "epsg:4326"
        tmp = ox.project_graph(g, CRS.from_user_input(3857))
        plot = figure(
            plot_height=plot_size,
            plot_width=plot_size,
            x_axis_type="mercator",
            y_axis_type="mercator",
            aspect_ratio=1,
            toolbar_location="below",
        )
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        plot.add_tile(tile_provider)
    else:
        tmp = (
            g  # projection not needed for toy networks, coordinates are plain
            # cartesian
        )
        plot = figure(
            plot_height=plot_size,
            plot_width=plot_size,
            aspect_ratio=1,
            toolbar_location="below",
        )
    plot.title.text = title
    max_width_bokeh, max_width_coords = get_max_edge_width(tmp, scaling, plot_size)
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
            ls, para_ls = __linestring_from_node_cords(
                [
                    [tmp.nodes[u]["x"], tmp.nodes[u]["y"]],
                    [tmp.nodes[v]["x"], tmp.nodes[v]["y"]],
                ],
                width_coords,
            )
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x = x2 + x1
            y = y2 + y1
            x_list.append(list(x))
            y_list.append(list(y))
    node_source = ColumnDataSource(
        data=dict(
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
    patch_ratio=8,
):
    nr_of_colors = len(traffic_cm)
    min_width_coords = max_width_coords / patch_ratio
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
            color = traffic_cm[-1]  # flow larger then capacity!
            flow = 0
        except KeyError:  # capacity or flow not defined
            color = traffic_cm[0]
            flow = 0
        colors.append(color)
        loaded = 0
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
        if "geometry" in data:
            ls = data["geometry"]
            assert isinstance(ls, LineString)
            para_ls = ls.parallel_offset(width_coords * 1)
        else:
            ls, para_ls = __linestring_from_node_cords(
                [
                    [g.nodes[u]["x"], g.nodes[u]["y"]],
                    [g.nodes[v]["x"], g.nodes[v]["y"]],
                ],
                width_coords,
            )

        try:
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x = x2 + x1
            y = y2 + y1
        except (
            AttributeError,
            NotImplementedError,
        ):  # Attributeerror: weird error due to i'll defined line
            # string .. - dig deeper if I have more time on my hands - probably an
            # error in osmnx line string
            # creation
            # Notimplemented Error - original Linestring is cut into multiple pieces
            # by parallel offset -
            # hence ls is MultiLineString - if the line string has very sharp corners
            # the offset will stop working
            # properly, we just use a straight Line connection in that case
            ls, para_ls = __linestring_from_node_cords(
                [
                    [g.nodes[u]["x"], g.nodes[u]["y"]],
                    [g.nodes[v]["x"], g.nodes[v]["y"]],
                ],
                width_coords,
            )
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x = x1 + x2
            y = y1 + y2
        x_list.append(list(x))
        y_list.append(list(y))

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


def __linestring_from_node_cords(coord_list, width_coords):
    ls = LineString(coord_list)
    return ls, ls.parallel_offset(1 * width_coords)
