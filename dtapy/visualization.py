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
import numpy as np
import networkx as nx
from bokeh.io import show, output_file, output_notebook
from bokeh.models import HoverTool, TapTool, OpenURL
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models.glyphs import Asterisk, Patches
from bokeh.models import CustomJS, Slider
import bokeh.plotting.figure as bk_figure
from bokeh.io import curdoc, show
from bokeh.layouts import row, widgetbox
from bokeh.models.widgets import Slider, TextInput
import numpy as np
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from shapely.geometry import LineString
from utilities import __create_green_to_red_cm
import osmnx as ox
from pyproj import CRS
from __init__ import results_folder
from dtapy.settings import parameters
from dtapy.core.time import SimulationTime
from dtapy.utilities import log

traffic_cm = __create_green_to_red_cm('hex')
default_plot_size = parameters.visualization.plot_size
default_notebook_plot_size = parameters.visualization.notebook_plot_size
default_max_links = parameters.visualization.max_links


def show_assignment(g: nx.DiGraph, flows, costs, time: SimulationTime, scaling=np.double(0.006), background_map=True,
                    title=None, plot_size=default_plot_size, osm_tap_tool=True, notebook=False,
                    max_links_visualized=default_max_links,
                    show_unloaded_links=False):
    """

    Parameters
    ----------
    osm_tap_tool
    notebook
    max_links_visualized
    show_unloaded_links
    flows
    costs
    g : nx.Digraph
    scaling : width of the widest link in relation to plot_size
    background_map : bool, show the background map
    title : str, plot title
    plot_size : height and width measurement in pixel

    Returns
    -------

    """
    plot = bk_figure(plot_height=plot_size,
                     plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                     aspect_ratio=1, toolbar_location='below')
    # adding different coordinate attribute names to comply with osmnx
    for _,_, data in g.edges.data():
        if 'x_coord' in data:
            data['x'] = data['x_coord']
            data['y'] = data['y_coord']
    for _,data in g.nodes.data():
        data['x'] = data['x_coord']
        data['y'] = data['y_coord']
    title=_check_title(title,g, 'assignment ')

    tmp = ox.project_graph(g, CRS.from_user_input(3857))  # from lan lot to web mercator
    _output(notebook, title, plot_size)  # only made to work inside of notebooks
    # TODO: possibly add standalone version of this plot for static html (likely without slider)
    if background_map:
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        plot.add_tile(tile_provider)
    # tmp = filter_links(tmp, max_links_visualized, show_unloaded_links, flows, costs)
    max_flow = np.max(flows)
    node_x = [x for _, x in tmp.nodes.data('x')]
    node_y = [y for _, y in tmp.nodes.data('y')]
    min_x, max_x, min_y, max_y = min(node_x), max(node_x), min(node_y), max(node_y)
    max_width_coords = scaling * (0.5 * (max_x - min_x) + 0.5 * (max_y - min_y))
    max_width_bokeh = plot_size * scaling

    # calculate all colors and coordinates for the different time dependent flows
    all_colors = []
    all_x = []
    all_y = []
    for t in range(time.tot_time_steps):
        c, x, y = _get_colors_and_coords(g, max_width_coords, max_flow, flows[t])
        all_colors.append(c)
        all_x.append(x)
        all_y.append(y)

    edge_source = _edge_cds(tmp, all_colors[0], flows[0], all_x[0],all_y[0], costs[0])
    node_source = _node_cds(tmp)

    plot.title.text = title

    edge_renderer = plot.add_glyph(edge_source,
                                   glyph=Patches(xs='x', ys='y', fill_color='color', line_color='color',
                                                 line_alpha=0.8))
    edge_tooltips = [(item, f'@{item}') for item in parameters.visualization.edge_keys if item != 'flow']
    edge_tooltips.append(('flow', '@flow{(0.0)}'))
    node_renderer = plot.add_glyph(node_source,
                                   glyph=Asterisk(x='x_coord', y='y_coord', size=max_width_bokeh * 3,
                                                  line_color="black",
                                                  line_width=max_width_bokeh / 5))
    node_tooltips = [(item, f'@{item}') for item in parameters.visualization.node_keys]

    edge_hover = HoverTool(show_arrow=False, tooltips=edge_tooltips, renderers=[edge_renderer])
    node_hover = HoverTool(show_arrow=False, tooltips=node_tooltips, renderers=[node_renderer])

    if osm_tap_tool:
        url = "https://www.openstreetmap.org/node/@ext_id/"
        nodetaptool = TapTool(renderers=[node_renderer])
        nodetaptool.callback = OpenURL(url=url)

    time_slider = Slider(start=0, end=10, value=0, step=1, title="time")
    # TODO: add color converter for dynamic

    # layout with multiple convergence plots
    # layout = row(
    #     plot,
    #     column(time_slider),
    # )
    plot.add_tools(node_hover, edge_hover, nodetaptool)

    text = TextInput(title="title", value='my Assignment plot')

    # Set up callbacks


    def update_title(attrname, old, new):
        plot.title.text = text.value

    def update_data(attrname, old, new):
        # Get the current slider values
        t = time_slider.value

        # Generate the new curve
        cur_flows = flows[t]
        cur_costs = costs[t]
        cur_color = all_colors[t]
        cur_x = all_x[t]
        cur_y = all_y[t]
        edge_source.data = {**edge_source.data, 'x_coord': cur_x, 'y_coord': cur_y, 'color': cur_color,
                            'flow': cur_flows, 'cost': cur_costs}
        ### I thought I might need a show() here, but it doesn't make a difference if I add one
        # show(layout)

    time_slider.on_change('value', update_data)

    # Set up layouts and add to document
    inputs = widgetbox(text, time_slider)
    layout = row(plot,
                 widgetbox(text, time_slider))
    output_notebook()

    def modify_doc(doc):
        doc.add_root(row(layout, width=800))
        doc.title = "Sliders"
        text.on_change('value', update_title)

    handler = FunctionHandler(modify_doc)
    app = Application(handler)
    show(app)
    # TODO: add update function that pushes to the notebook for different times ..
    # TODO: add animation that can be interrupted


def show_demand(g, plot_size=1300, notebook=False, title=None):
    _check_title(title, g, plot_type='Desire Lines ')
    if notebook:
        output_notebook(hide_banner=True)
        plot_size = 600
    else:
        output_file(results_folder + f'/{title}.html')
    plot = figure(plot_height=plot_size,
                  plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                  aspect_ratio=1, toolbar_location='below')
    tmp = ox.project_graph(g, CRS.from_user_input(3857))
    # edge_renderer = plot.add_glyph(edge_source,
    #                                glyph=Patches(xs='x', ys='y', fill_color='green', line_color='green',
    #                                              line_alpha=0.8))
    # edge_tooltips = [('flow', '@flow{(0.0)}')]
    # node_renderer = plot.add_glyph(node_source,
    #                                glyph=Asterisk(x='x', y='y', size=max_width_bokeh * 3, line_color="black",
    #                                               line_width=max_width_bokeh / 5))
    # node_tooltips = [(item, f'@{item}') for item in parameters.visualization.node_keys]


def filter_links(g: nx.DiGraph, max_links_visualized, show_unloaded_links, flows, costs):
    """
    returns filtered network graph either excluding unloaded edges or/and only including the most loaded edges across all
    time steps.
    Parameters
    ----------
    g: nx.MultiDiGraph
    max_links_visualized: maximum number of links to show
    show_unloaded_links: whether or not to include links that do not have loads
    flows: array of flows
    costs: array of costs

    Returns
    -------
    filtered g, flows and costs
    """
    if not show_unloaded_links:
        loaded_links = np.argwhere(np.sum(flows, axis=1) > 0)
        edges = [(u, v, k) for u, v, k, data in g.edges.data(keys=True) if data['link_id'] in loaded_links]
        g = g.edge_subgraph(edges)
        flows = flows[:, loaded_links]
        costs = costs[:, loaded_links]
    if g.number_of_edges() > max_links_visualized:
        links_to_show = np.argsort(np.sum(flows, axis=1))[:max_links_visualized]
        edges = [(u, v, k) for u, v, k, data in g.edges.data(keys=True) if data['link_id'] in links_to_show]
        flows = flows[:, links_to_show]
        costs = costs[:, links_to_show]
        g = g.edge_subgraph(edges)
    return g, flows, costs


# def filter_time(time: SimulationTime, flows, costs):
#     last = np.max(np.argwhere(np.sum(flows, axis=0) > 0))
#     flows = flows[:last, :]
#     costs = costs[:last, :]
#     new_time = SimulationTime(time.start, last * time.step_size, time.step_size)
#     return new_time


def show_convergence(g: nx.DiGraph, notebook=False):
    assert 'iterations' in g.graph
    assert 'gaps' in g.graph
    assert 'name' in g.graph
    name = g.graph['name']
    title = f'convergence {name}'
    if notebook:
        output_notebook()
    else:
        output_file(results_folder + f"/{title}.html")
    gaps = g.graph['gaps']
    iterations = np.arange(len(gaps))
    p = figure(plot_width=400, plot_height=400, title=title, x_axis_label='Iterations', y_axis_label='Gap')
    p.line(iterations, gaps, line_width=2)
    p.circle(iterations, gaps, fill_color="white", size=8)
    p.add_tools(HoverTool())
    show(p)


def _node_cds(g, visualization_keys=parameters.visualization.node_keys):
    node_dict = dict()
    visualization_keys.append('x')
    visualization_keys.append('y')
    for attr_key in visualization_keys:
        values = [node_attr[attr_key] if attr_key in node_attr.keys() else 'None'
                  for _, node_attr in g.nodes(data=True)]
        node_dict[attr_key] = values
    return ColumnDataSource(data=node_dict)


def _edge_cds(g,color, flow, x,y, cost, visualization_keys=parameters.visualization.edge_keys):
    # TODO: add dependence on cost and flow array.
    edge_dict = dict()
    for attr_key in visualization_keys:
        values = [edge_attr[attr_key] if attr_key in edge_attr.keys() else 'None'
                  for _, _, edge_attr in sorted(g.edges(data=True), key=lambda t: t[2]['link_id'])]
        edge_dict[attr_key] = values
    edge_dict['color'] = color
    edge_dict['flow'] = flow
    edge_dict['x'] = x
    edge_dict['y'] = y
    edge_dict['cost'] =cost
    return ColumnDataSource(data=edge_dict)


def _get_colors_and_coords(g, max_width_coords, max_flow, flows):
    nr_of_colors = len(traffic_cm)
    min_width_coords = max_width_coords / 10
    colors = []
    x_list = []
    y_list = []
    for u, v, data in g.edges(data=True):
        try:
            try:
                print()
                color = traffic_cm[np.int(np.round(flows[data['link_id']] / data['capacity'] * nr_of_colors))]
            except IndexError:
                color = traffic_cm[-1]  # flow larger then capacity!
            except KeyError:  # capacity or flow not defined
                color = traffic_cm[0]
            colors.append(color)
            width_coords = min_width_coords + (max_width_coords - min_width_coords) * (
                    flows[data['link_id']] / max_flow)
            # width_bokeh = min_width_bokeh + (max_width_bokeh - min_width_bokeh) * (data['flow'] / max_flow)
        except KeyError:  # flow not defined.., no width scaling possible
            width_coords = min_width_coords
            # width_bokeh = min_width_bokeh
        # edge_dict['width'].append(width_bokeh)
        if 'geometry' in data:
            ls = data['geometry']
            assert isinstance(ls, LineString)
            para_ls = ls.parallel_offset(width_coords * 1)
        else:
            ls, para_ls = __linestring_from_node_cords(
                [[g.nodes[u]['x'], g.nodes[u]['y']], [g.nodes[v]['x'], g.nodes[v]['y']]],
                width_coords)

        try:
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x = x2 + x1
            y = y2 + y1
        except (AttributeError, NotImplementedError) as e:  # Attributeerror: weird error due to i'll defined line
            # string .. - dig deeper if I have more time on my hands - probably an error in osmnx line string
            # creation
            # Notimplemented Error - original Linestring is cut into multiple pieces by parallel offset -
            # hence ls is MultiLineString - if the line string has very sharp corners the offset will stop working
            # properly, we just use a straight Line connection in that case
            ls, para_ls = __linestring_from_node_cords(
                [[g.nodes[u]['x'], g.nodes[u]['y']], [g.nodes[v]['x'], g.nodes[v]['y']]],
                width_coords)
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x = x1 + x2
            y = y1 + y2
        x_list.append(list(x))
        y_list.append(list(y))
    return colors, x_list, y_list


def __linestring_from_node_cords(coord_list, width_coords):
    ls = LineString(coord_list)
    return ls, ls.parallel_offset(1 * width_coords)


def _check_title(title, tmp, plot_type: str):
    if title is None:
        try:
            tmp.graph['name'] = tmp.graph['name'].strip('_UTM')
            title = plot_type + ' in ' + tmp.graph['name']
            time = tmp.graph.get('time', None)
            time_str = 'at' + str(time)
            if time is not None:
                title = title + time_str
        except KeyError:
            # no name provided ..
            title = plot_type + ' ' + '... provide city name in graph and it will show here..'
    return title


def _output(notebook: bool, title, plot_size):
    if notebook:
        output_notebook(hide_banner=True)
        if not plot_size:
            plot_size = default_notebook_plot_size
    else:
        if not plot_size:
            plot_size = default_plot_size
        output_file(results_folder + f'/{title}.html')
