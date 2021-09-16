#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#

import numpy as np
from osmnx.distance import get_nearest_edges, great_circle_vec
import networkx as nx
from dyntapy.dta.core.filters.helbing_treiber import treiber_helbing_asm


class Detectors():
    def __init__(self, x_coord, y_coord, speeds: np.ndarray=None, densities: np.ndarray = None, flows: np.ndarray = None,
                 g: nx.MultiDiGraph = None):
        """

        Parameters
        ----------
        g : nx.MultiDiGraph - GMNS conform
        x_coord : longitude of all detectors
        y_coord : latitude of all detectors
        densities : Array, tot_detectors x tot_time_intervals
        speeds : Array, tot_detectors x tot_time_intervals
        flows : Array, tot_detectors x tot_time_intervals
        """
        self.tot_detectors = speeds.shape[0]
        self.tot_time_intervals = speeds.shape[1]
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.densities = densities
        self.speeds = speeds
        self.flows = flows
        self.detector_ids = np.arange(self.tot_detectors)
        self.link_ids = self.get_link_ids(g)
        self.per_link = []
        for _ in g.number_of_edges():
            self.per_link.append([])  # each link may have multiple detectors
        for detector_id, link_id in self.link_ids:
            self.per_link[link_id].append(detector_id)
        self.g = g

    def get_link_ids(self, g):
        """
        gets the link ids for all of the detectors by finding the closest edges for each.
        Parameters
        ----------
        g: nx.MultiDiGraph, GMNS conform

        Returns
        -------
        array of matched link_ids
        """
        for u, v, d in g.edges():
            # different coordinate naming convention in osmnx
            d['x'] = d['x_coord']
            d['y'] = d['y_coord']
        edges = get_nearest_edges(g, self.x_coord, self.y_coord, method='ball_tree')
        for u, v, d in g.edges():
            # clean up graph
            del d['x']
            del d['y']
        return np.array([g[u][v][k]['link_id'] for u, v, k in edges])

    def set_link_id(self, detector_id, new_link_id):
        """
        manual method to override detector matching for a specific detector
        Parameters
        ----------
        detector_id: int
        new_link_id: int

        Returns
        -------

        """
        self.link_ids[detector_id] = new_link_id

    def get_xt_data(self, path):
        """

        Parameters
        ----------
        path : list of link_ids that form the path
        Returns
        -------

        """
        detector_ids = []
        detector_locations_on_path = []
        X = np.float(0)  # len of X axis in plot
        sorted_edges = sorted(self.g.edges(data=True), key=lambda t: t[2]['link_id'])
        for link in path:
            u, v, d = sorted_edges[link]
            X += d['length']

            for detector_id in self.per_link[link]:
                detector_ids.append(detector_id)
        pass
        #treiber_helbing_asm()
