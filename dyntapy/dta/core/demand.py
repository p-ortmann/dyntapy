#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
import numpy as np
from numba import uint32, njit
from collections import OrderedDict
from numba.core.types.containers import ListType
from numba.experimental import jitclass
from dyntapy.datastructures.csr import f32csr_type, UI32CSRMatrix
from dyntapy.sta.demand import Demand


try:
    spec_simulation = [('next', Demand.class_type.instance_type),
                       ('demands', ListType(Demand.class_type.instance_type)),
                       ('__time_step', uint32),
                       ('tot_time_steps', uint32),
                       ('all_active_destinations', uint32[:]),
                       ('all_active_destination_links', uint32[:]),
                       ('all_active_origins', uint32[:]),
                       ('all_centroids', uint32[:]),
                       ('tot_centroids', uint32),
                       ('tot_active_destinations', uint32),
                       ('tot_active_origins', uint32),
                       ('loading_time_steps', uint32[:])]
except Exception:
    # numba disabled
    spec_simulation=None

@jitclass(spec_simulation)
class InternalDynamicDemand(object):
    def __init__(self, demands, tot_time_steps, tot_centroids, in_links: UI32CSRMatrix):
        self.demands = demands
        self.next = demands[0]
        self.loading_time_steps = _get_loading_time_steps(demands)
        # time step traffic is loaded into the network
        self.all_active_destinations = get_all_destinations(demands)
        self.all_active_destination_links = get_destination_links(self.all_active_destinations, in_links)
        self.all_active_origins = get_all_origins(demands)
        self.tot_active_origins = self.all_active_origins.size
        self.tot_active_destinations = self.all_active_destinations.size
        self.all_centroids = np.arange(tot_centroids, dtype=np.uint32)  # for destination/origin based labels
        self.tot_time_steps = np.uint32(tot_time_steps)
        self.tot_centroids = np.uint32(tot_centroids)

    def is_loading(self, t):
        _ = np.argwhere(self.loading_time_steps == t)
        if _.size == 1:
            return True
        elif _.size > 1:
            raise Exception('ValueError, multiple StaticDemand objects with identical time label')
        else:
            return False

    def get_demand(self, t):
        _id = np.argwhere(self.loading_time_steps == t)[0][0]
        return self.demands[_id]


@njit(cache=True)
def _get_loading_time_steps(demands):
    loading = np.empty(len(demands), dtype=np.uint32)
    for _id, demand in enumerate(demands):
        demand: Demand
        t = demand.time_step
        loading[_id] = np.uint32(t)
    return loading


@njit
def get_all_destinations(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].destinations
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: Demand
        current = np.concatenate((demand.destinations, previous))
        previous = current
    return np.unique(current)


@njit
def get_all_origins(demands):
    if len(demands) < 1:
        raise AssertionError
    previous = demands[0].origins
    if len(demands) == 1:
        return previous
    for demand in demands[1:]:
        demand: Demand
        current = np.concatenate((demand.origins, previous))
        previous = current
    return np.unique(current)

@njit(cache=True)
def get_destination_links(destinations: np.ndarray, in_links:UI32CSRMatrix):
    """
    Parameters
    ----------
    destinations : destinations to get links for
    in_links : CSRMatrix, in_links for all nodes in the network. Assumes that every centroid only has one in_link.

    Returns
    -------
    array containing the corresponding connector for each destination

    """
    destinations_link= np.empty(destinations.size, dtype=np.uint32)
    for d_id, destination in enumerate(destinations):
        assert in_links.get_nnz(destination).size==1
        for link in in_links.get_nnz(destination):
            destinations_link[d_id]=link
    return destinations_link
