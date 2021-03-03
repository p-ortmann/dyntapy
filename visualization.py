#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
import networkx as nx
from bokeh.io import show, output_file, output_notebook
from bokeh.models import HoverTool, TapTool, OpenURL
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.glyphs import Asterisk, Patches
from bokeh.models import CustomJS, Slider
from shapely.geometry import LineString
from utilities import __create_green_to_red_cm
import osmnx as ox
from pyproj import CRS
from __init__ import results_folder
from settings import parameters

traffic_cm = __create_green_to_red_cm('hex')
default_plot_size = parameters.visualization.plot_size
default_notebook_plot_size = parameters.visualization.notebook_plot_size


def show_assignment(g: nx.DiGraph,flows, costs, scaling=np.double(0.006), background_map=True,
                    title=None, plot_size=default_plot_size, osm_tap_tool=True, notebook=False, max_links_visualized=None,
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
    tmp = ox.project_graph(g, CRS.from_user_input(3857))
    title = _check_title(title, tmp, 'assignment')
    _output(notebook, title)
    plot = figure(plot_height=plot_size,
                  plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                  aspect_ratio=1, toolbar_location='below')

    if max_links_visualized is None:
        max_links_visualized = parameters.visualization.max_links
    tmp = filter_links(tmp, max_links_visualized, show_unloaded_links)
    max_flow = np.max(flows)

    node_x = [x for _, x in tmp.nodes.data('x_coord')]
    node_y = [y for _, y in tmp.nodes.data('y_coord')]
    min_x, max_x, min_y, max_y = min(node_x), max(node_x), min(node_y), max(node_y)
    max_width_coords = scaling * (0.5 * (max_x - min_x) + 0.5 * (max_y - min_y))
    max_width_bokeh = plot_size * scaling

    edge_source = _edge_cds(tmp, max_width_coords, max_flow,flows)
    node_source = _node_cds(tmp)
    _background_map(background_map, plot)
    plot.title.text = title

    edge_renderer = plot.add_glyph(edge_source,
                                   glyph=Patches(xs='x_coord', ys='y_coord', fill_color='color', line_color='color',
                                                 line_alpha=0.8))
    edge_tooltips = [(item, f'@{item}') for item in parameters.visualization.edge_keys if item != 'flow']
    edge_tooltips.append(('flow', '@flow{(0.0)}'))
    node_renderer = plot.add_glyph(node_source,
                                   glyph=Asterisk(x='x_coord', y='y_coord', size=max_width_bokeh * 3, line_color="black",
                                                  line_width=max_width_bokeh / 5))
    node_tooltips = [(item, f'@{item}') for item in parameters.visualization.node_keys]

    edge_hover = HoverTool(show_arrow=False, tooltips=edge_tooltips, renderers=[edge_renderer])
    node_hover = HoverTool(show_arrow=False, tooltips=node_tooltips, renderers=[node_renderer])

    if osm_tap_tool:
        url = "https://www.openstreetmap.org/node/@ext_id/"
        nodetaptool = TapTool(renderers=[node_renderer])
        nodetaptool.callback = OpenURL(url=url)

    time_slider = Slider(start=0, end=10, value=0, step=1, title="time")
    #TODO: add color converter for dynamic
    callback = CustomJS(
        args=dict(source=edge_source, time=time_slider, dynamic_x=None, dynamic_y=None, dynamic_color=None,
                  dynamic_flow=flows),
        code="""
        const data = source.data;
        const t = time.value;
        data['flow'] = dynamic_flow[t]
        data['x'] = dynamic_x[t]
        data['y'] = dynamic_y[t]
        data['color'] = dynamic_color[t]
        source.change.emit();
    """)
    time_slider.js_on_change('value', callback)
    # layout with multiple convergence plots
    layout = row(
        plot,
        column(time_slider),
    )
    plot.add_tools(node_hover, edge_hover, nodetaptool)

    show(plot)


def show_demand(g, plot_size=1300, notebook=False):
    _check_title(t)
    if notebook:
        output_notebook(hide_banner=True)
        plot_size = 600
    else:
        output_file(results_folder + f'/{title}.html')
    plot = figure(plot_height=plot_size,
                  plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                  aspect_ratio=1, toolbar_location='below')
    tmp = ox.project_graph(g, CRS.from_user_input(3857))
    edge_renderer = plot.add_glyph(edge_source,
                                   glyph=Patches(xs='x', ys='y', fill_color='green', line_color='green',
                                                 line_alpha=0.8))
    edge_tooltips = [('flow', '@flow{(0.0)}')]
    node_renderer = plot.add_glyph(node_source,
                                   glyph=Asterisk(x='x', y='y', size=max_width_bokeh * 3, line_color="black",
                                                  line_width=max_width_bokeh / 5))
    node_tooltips = [(item, f'@{item}') for item in parameters.visualization.node_keys]



def filter_links(g: nx.DiGraph, max_links_visualized, show_unloaded_links):
    if g.number_of_edges() > max_links_visualized:
        tmp = sorted(((np.max(data['flow']) / data['capacity'], u, v) for u, v, data in g.edges.data()),
                     reverse=True)
        if not show_unloaded_links:
            edges = [(u, v) for val, u, v in tmp[:max_links_visualized] if val > 0.00001]
        else:
            edges = [(u, v) for val, u, v in tmp[:max_links_visualized]]
        return g.edge_subgraph(edges)
    else:
        return g


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


def _node_cds(g, visualization_keys = parameters.visualization.node_keys):
    node_dict = dict()
    for attr_key in visualization_keys:
        values = [node_attr[attr_key] if attr_key in node_attr.keys() else 'None'
                  for _, node_attr in g.nodes(data=True)]
        node_dict[attr_key] = values
    return ColumnDataSource(data=node_dict)


def _edge_cds(g, max_width_coords, max_flow, visualization_keys= parameters.visualization.edge_keys):
    #TODO: add dependence on cost and flow array.
    edge_dict = dict()
    for attr_key in visualization_keys:
        values = [edge_attr[attr_key] if attr_key in edge_attr.keys() else 'None'
                  for _, _, edge_attr in g.edges(data=True)]
        edge_dict[attr_key] = values

    nr_of_colors = len(traffic_cm)
    min_width_coords = max_width_coords / 10
    edge_dict['x'], edge_dict['y'], edge_dict['color'] = [], [], [],
    for u, v, data in g.edges(data=True):
        try:
            try:
                color = traffic_cm[np.int(round(data['flow'] / data['capacity'] * nr_of_colors))]
            except IndexError:
                color = traffic_cm[-1]  # flow larger then capacity!
            except KeyError:  # capacity or flow not defined
                color = traffic_cm[0]
            edge_dict['color'].append(color)
            width_coords = min_width_coords + (max_width_coords - min_width_coords) * (data['flow'] / max_flow)
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
        edge_dict['x'].append(list(x))
        edge_dict['y'].append(list(y))
    return ColumnDataSource(data=edge_dict)


def __linestring_from_node_cords(coord_list, width_coords):
    ls = LineString(coord_list)
    return ls, ls.parallel_offset(1 * width_coords)


def _check_title(title, tmp, plot_type: str):
    if title is None:
        try:
            tmp.graph['name'] = tmp.graph['name'].strip('_UTM')
            title = plot_type + ' ' + tmp.graph['name']
        except KeyError:
            # no name provided ..
            title = plot_type + ' ' + '... provide city name in graph and it will show here..'
    return title


def _output(notebook: bool, title, plot_size):
    if notebook:
        output_notebook(hide_banner=True)
        plot_size = 600
    else:
        output_file(results_folder + f'/{title}.html')


def _background_map(background_map, plot):
    if background_map:
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        plot.add_tile(tile_provider)
