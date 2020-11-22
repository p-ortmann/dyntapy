#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
try:
    import win32com.client
    import os
    from glob import glob1
    import networkx as nx
    from pyproj import Proj, transform
    from settings import data_folder


    def visum_to_nx(name_of_city, path=None):
        visum = __dispatch_visum(path)
        g = __get_network_data(visum)
        g.graph['name']=name_of_city
        return g


    def __dispatch_visum(path):
        if path is None:
            if os.path.isdir(data_folder):
                ver_files = glob1(data_folder, '*.ver')
                if len(ver_files) != 1:
                    raise FileNotFoundError('no (or multiple) .ver files found in data folder')
                else:
                    ver_file = os.path.join(data_folder, ver_files[0])
                    visum = win32com.client.Dispatch('Visum.Visum')
                    path = os.path.join(os.getcwd(), ver_file)
                    visum.LoadVersion(path)
            else:
                raise NotADirectoryError('data folder not found! could not check for .ver file')
        else:
            visum = win32com.client.Dispatch('Visum.Visum')
            try:
                visum.LoadVersion(path)
            except BaseException as e:
                print(e.args)
                print('no valid path name')
                raise WindowsError

        return visum


    def __get_network_data(visum: win32com.client.CDispatch):
        no = [int(val) for _, val in list(visum.Net.Links.GetMultiAttValues('No', True))]
        from_node = [int(val) for _, val in list(visum.Net.Links.GetMultiAttValues('FromNodeNo', True))]
        to_node = [int(val) for _, val in list(visum.Net.Links.GetMultiAttValues('ToNodeNo', True))]
        caps = [val for _, val in list(visum.Net.Links.GetMultiAttValues('CapPrt', True))]
        lengths = [val for _, val in list(visum.Net.Links.GetMultiAttValues('Length', True))]
        maxspeeds = [val for _, val in list(visum.Net.Links.GetMultiAttValues('V0PrT', True))]
        link_data = tuple([(u, v, {'capacity': cap,
                                   'from_node_visum': u,
                                   'to_node_visum': v,
                                   'link_id_visum': link_id,
                                   'length': length,
                                   'maxspeed': speed}) for u, v, link_id, cap, length, speed in
                           zip(from_node, to_node, no, caps, lengths, maxspeeds)])
        node_ids = [int(val) for _, val in list(visum.Net.Nodes.GetMultiAttValues('No', True))]
        node_xs_raw = [val for _, val in list(visum.Net.Nodes.GetMultiAttValues('XCoord', True))]
        node_ys_raw = [val for _, val in list(visum.Net.Nodes.GetMultiAttValues('YCoord', True))]
        in_proj, out_proj = Proj('epsg:31370'), Proj('epsg:4326')  # from belge_lambert to lat lon (epsg4326)
        node_xs, node_ys = transform(in_proj, out_proj, node_xs_raw, node_ys_raw)
        node_data = tuple([(u, {'lon': x, 'lat': y}) for u, x, y in zip(node_ids, node_xs, node_ys)])
        g = nx.DiGraph()
        g.add_nodes_from(node_data)
        g.add_edges_from(link_data)
        return g
except ImportError:
    print('unix like system, visum can not be interfaced with')
