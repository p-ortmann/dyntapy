#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
import networkx as nx
from bokeh.io import show, output_file, output_notebook
from bokeh.models import HoverTool, CustomJS, TapTool, OpenURL, Label
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.glyphs import MultiLine, Circle, X, Asterisk, Patches
from shapely.geometry import LineString
from stapy.settings import visualization_keys_edges, visualization_keys_nodes
from stapy.utilities import log, __create_green_to_red_cm
from stapy.assignment import StaticAssignment
import osmnx as ox
from pyproj import CRS

traffic_cm = __create_green_to_red_cm('hex')


def plot_network(g: nx.DiGraph, scaling=np.double(0.006), background_map=True,
                 title=None, plot_size=1300,
                 mode='assignment', iteration=None, notebook=False, max_links_visualized=None, show_unloaded_links=False):
    """

    Parameters
    ----------
    plot_nodes : bool, whether nodes should also be plotted
    g : nx.Digraph
    scaling : width of the widest link in relation to plot_size
    background_map : bool, show the background map
    title : str, plot title
    show_iterations : bool
    plot_size : height and width measurement in pixel
    with_assignment : bool, indicates if each edge has flow variable

    Returns
    -------

    """
    tmp = nx.MultiDiGraph(g)
    tmp = ox.project_graph(tmp, CRS.from_user_input(3857))
    tmp.graph['name'] = tmp.graph['name'].strip('_UTM')
    g=nx.DiGraph(tmp)
    if notebook:
        plot_size = 900
    assert mode in ['assignment', 'desire lines', 'deleted elements']
    plot = figure(plot_height=plot_size,
                  plot_width=plot_size, x_axis_type="mercator", y_axis_type="mercator",
                  aspect_ratio=1, toolbar_location='below')
    if iteration is not None:
        assert isinstance(iteration, int)
        assert 'iterations' in g.graph
        assert 'gaps' in g.graph
        flows = g.graph['iterations'][iteration]
        gap = g.graph['gaps'][iteration]
        citation = Label(x=30, y=30, x_units='screen', y_units='screen',
                         text=f'Iteration {iteration}, Remaining Gap: {gap} ', render_mode='css',
                         border_line_color='black', border_line_alpha=1.0,
                         background_fill_color='white', background_fill_alpha=1.0)
        plot.add_layout(citation)
        for flow, (_, _, data) in zip(flows, g.edges.data()):
            data['flow'] = flow
    if mode in ['assignment', 'desire lines']:
        if max_links_visualized is None:
            from settings import max_links_visualized
        g = filter_links(g, max_links_visualized, show_unloaded_links)
        max_flow = max([float(f) for _, _, f in g.edges.data('flow') if f is not None])
    else:
        max_flow = 0

    node_x = [x for _, x in g.nodes.data('x')]
    node_y = [y for _, y in g.nodes.data('y')]
    min_x, max_x, min_y, max_y = min(node_x), max(node_x), min(node_y), max(node_y)
    max_width_coords = scaling * (0.5 * (max_x - min_x) + 0.5 * (max_y - min_y))
    max_width_bokeh=plot_size*scaling

    edge_source = _edge_cds(g, max_width_coords, max_flow)
    node_source = _node_cds(g, mode == 'assignment')

    if background_map:
        tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
        plot.add_tile(tile_provider)
    if title == None:
        try:
            title = mode + ' ' + g.graph['name']
        except KeyError:
            # no name provided ..
            title = mode + ' ' + '... provide city name in graph and it will show here..'
    plot.title.text = title

    if mode == 'assignment':
        #edge_renderer = plot.add_glyph(edge_source,
         #                              glyph=MultiLine(xs='x', ys='y', line_width='width', line_color='color',
          #                                             line_alpha=0.8))
        edge_renderer = plot.add_glyph(edge_source,
                                      glyph=Patches(xs='x', ys='y', fill_color='color', line_color='color',
                                                     line_alpha=0.8))
        edge_tooltips = [(item, f'@{item}') for item in visualization_keys_edges if item != 'flow']
        edge_tooltips.append(('flow', '@flow{(0.0)}'))
        node_renderer = plot.add_glyph(node_source,
                                       glyph=Asterisk(x='x', y='y', size=max_width_bokeh * 3, line_color="black",
                                                      line_width=max_width_bokeh / 5))
        node_tooltips = [(item, f'@{item}') for item in visualization_keys_nodes]
    if mode == 'desire lines':
        edge_renderer = plot.add_glyph(edge_source,
                                       glyph=Patches(xs='x', ys='y', fill_color='green', line_color='green',
                                                     line_alpha=0.8))
        edge_tooltips = [('flow', '@flow{(0.0)}')]
        node_renderer = plot.add_glyph(node_source,
                                       glyph=Asterisk(x='x', y='y', size=max_width_bokeh * 3, line_color="black",
                                                      line_width=max_width_bokeh / 5))
        node_tooltips = [(item, f'@{item}') for item in visualization_keys_nodes]
    if mode == 'deleted elements':
        edge_renderer = plot.add_glyph(edge_source,
                                       glyph=Patches(xs='x', ys='y', fill_color='red', line_color='red',
                                                     line_alpha=0.8))
        tmp = ['name', 'highway', 'length', 'maxspeed']
        edge_tooltips = [(item, f'@{item}') for item in tmp if item != 'flow']
        node_renderer = plot.add_glyph(node_source,
                                       glyph=Circle(x='x', y='y', size=max_width_bokeh, line_color="red",
                                                    fill_color="black", line_width=max_width_bokeh / 8))
        tmp = ['osmid', 'x', 'y']
        node_tooltips = [(item, f'@{item}') for item in tmp]

    edge_hover = HoverTool(show_arrow=False, tooltips=edge_tooltips, renderers=[edge_renderer])
    node_hover = HoverTool(show_arrow=False, tooltips=node_tooltips, renderers=[node_renderer])
    url = "https://www.openstreetmap.org/node/@osmid/"
    nodetaptool = TapTool(renderers=[node_renderer])
    nodetaptool.callback = OpenURL(url=url)
    url = "https://www.openstreetmap.org/way/@osmid/"
    edgetaptool = TapTool(renderers=[edge_renderer])
    edgetaptool.callback = OpenURL(url=url)
    plot.add_tools(node_hover, edge_hover, edgetaptool, nodetaptool)
    if notebook:
        output_notebook()
    else:
        output_file(f"data/{title}.html")
    show(plot)


def show_desire_lines(g: nx.DiGraph, od_matrix, plot_size=1300, notebook=False):
    obj = StaticAssignment(g, od_matrix)
    od_flow_graph = obj.construct_demand_graph()
    plot_network(od_flow_graph, plot_size=plot_size, notebook=notebook, mode='desire lines')


def filter_links(g: nx.DiGraph, max_links_visualized, show_unloaded_links):
    if g.number_of_edges() > max_links_visualized:
        tmp = sorted(((data['flow'] / data['capacity'], u, v) for u, v, data in g.edges.data()),
                     reverse=True)
        if not show_unloaded_links:
            edges = [(u, v) for val, u, v in tmp[:max_links_visualized] if val>0.01]
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
        output_file(f"data/{title}.html")
    gaps = g.graph['gaps']
    iterations = np.arange(len(gaps))
    output_file("multiple.html")
    p = figure(plot_width=400, plot_height=400, title=title, x_axis_label='Iterations', y_axis_label='Gap')
    p.line(iterations, gaps, line_width=2)
    p.circle(iterations, gaps, fill_color="white", size=8)
    p.add_tools(HoverTool())
    show(p)


def _node_cds(g, with_assignment: bool):
    node_dict = dict()
    if with_assignment:
        nodes = [u for u, data in g.nodes.data() if 'origin' in data or 'destination' in data]
        g = g.subgraph(nodes)
    for attr_key in visualization_keys_nodes:
        values = [node_attr[attr_key] if attr_key in node_attr.keys() else None
                  for _, node_attr in g.nodes(data=True)]
        node_dict[attr_key] = values
    return ColumnDataSource(data=node_dict)


def _edge_cds(g, max_width_coords, max_flow):
    edge_dict = dict()
    for attr_key in visualization_keys_edges:
        values = [edge_attr[attr_key] if attr_key in edge_attr.keys() else None
                  for _, _, edge_attr in g.edges(data=True)]
        edge_dict[attr_key] = values
    edge_dict['compressed_osm_ids'] = []
    for it,id in enumerate(edge_dict['osmid']):
        if isinstance(id, list):
            edge_dict['compressed_osm_ids'].append(id[1:])
            edge_dict['osmid'][it] = id[0]
        else:
            edge_dict['compressed_osm_ids'].append([-1])


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
            #width_bokeh = min_width_bokeh + (max_width_bokeh - min_width_bokeh) * (data['flow'] / max_flow)
        except KeyError:  # flow not defined.., no width scaling possible
            width_coords = min_width_coords
            #width_bokeh = min_width_bokeh
        #edge_dict['width'].append(width_bokeh)
        if 'geometry' in data:
            ls = data['geometry']
            assert isinstance(ls, LineString)
            para_ls = ls.parallel_offset(width_coords * 1)
        else:
            ls, para_ls = __linestring_from_node_cords([[g.nodes[u]['x'], g.nodes[u]['y']], [g.nodes[v]['x'], g.nodes[v]['y']]],
                                              width_coords)

        try:
            (x1, y1, x2, y2) = ls.xy + para_ls.xy
            x=x2+x1
            y=y2 +y1
        except (AttributeError, NotImplementedError) as e:  # Attributeerror: weird error due to i'll defined line
            # string .. - dig deeper if I have more time on my hands - probably an error in osmnx line string
            # creation
            # Notimplemented Error - original Linestring is cut into multiple pieces by parallel offset -
            # hence ls is MultiLineString - if the line string has very sharp corners the offset will stop working
            # properly, we just use a straight Line connection in that case
            ls, para_ls = __linestring_from_node_cords([[g.nodes[u]['x'], g.nodes[u]['y']], [g.nodes[v]['x'], g.nodes[v]['y']]],
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




    # for x1, y1, x2, y2, width in zip(x1s, y1s, x2s, y2s, 0.5*widths):
    #     atan_x = x2 - x1
    #     atan_y = y2 - y1
    #     if atan_y >= 0:
    #         if atan_x >= 0:
    #             dx = width * cos(atan2(atan_x, atan_y))
    #             dy = width * sin(atan2(atan_x, atan_y))
    #         else:
    #             dx = width * sin(atan2(atan_x, atan_y))
    #             dy = width * cos(atan2(atan_x, atan_y))
    #     else:
    #         if atan_x >= 0:
    #             dx = width * sin(atan2(atan_x, atan_y))
    #             dy = width * cos(atan2(atan_x, atan_y))
    #         else:
    #             dx = width * cos(atan2(atan_x, atan_y))
    #             dy = width * sin(atan2(atan_x, atan_y))
    #             new_x.append([x1 + dx,x2 + dx])
    #             new_y.append([y1 + dy,y2 + dy])
    #     return new_x, new_y



