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
"""
    The DiGraph format of networkx is very flexible in its formatting.
    We describe below the semantics and labelling requirements for the DiGraph in
    order to use the assignments in dyntapy.

    All nodes and links need to have fully specified attributes for a subset of
    the General Modeling Network Specification (GMNS), see
    https://github.com/zephyr-data-specs/GMNS.


    For the links we need:
    'from_node_id', 'to_node_id', 'link_id', 'lanes',
    'capacity', 'length', 'free_speed'

    optional are:

    'geometry': shapely.geometry.LineString
        if it is not set we assume a straight line
        between the two nodes.

    'link_type': int,
        set to 1 for source connectors, -1 for sinks, 0 otherwise.

    'connector': bool,
        True if the link is a connector


    For the nodes:
    'node_id', 'x_coord', 'y_coord', 'ctrl_type', 'node_type'

    optional are:

    'centroid': bool,
        True if the node is a centroid

    The inclusion of the 'link_type', 'connector' and 'centroid' attributes are
    deviations from GMNS.

    The graph's nodes and edges need to be labelled consecutively and starting
    from 0.
    Many of the assignment algorithms also implicitly assume that the graph is strongly
    connected.
    This can be verified with networkx:

    >>> networkx.strongly_connected_components(g)

    If there is only a single element returned as the list of components, the graph
    is strongly connected.

    All of these requirements are met if dyntapy's functions for extracting the network
    from OpenStreetMap are used.

    See Also
    --------

    dyntapy.supply_data.road_network_from_place

    dyntapy.demand_data.add_centroids

    dyntapy.demand_data.add_connectors

    dyntapy.supply_data.relabel_graph



"""
import networkx
import numpy as np

import dyntapy._context
from dyntapy.demand import (
    InternalDynamicDemand,
    build_internal_static_demand,
    build_internal_dynamic_demand,
)
from dyntapy.demand_data import _check_centroid_connectivity
from dyntapy.dta.aon import aon
from dyntapy.dta.i_ltm_aon import i_ltm_aon
from dyntapy.dta.incremental_assignment import incremental
from dyntapy.sta.dial_b import dial_b
from dyntapy.sta.msa import msa_flow_averaging
from dyntapy.sta.uncongested_dial import sun
from dyntapy.supply_data import build_network
from dyntapy.utilities import log
from dyntapy.results import StaticResult, get_skim, DynamicResult


class DynamicAssignment:
    """This class stores all the information needed for the assignment itself.
    upon initialisation both the network and dynamic demand are transformed into
    internal representations.
    """

    def __init__(
        self,
        network,
        dynamic_demand,
        simulation_time,
    ):
        """

        Parameters
        ----------

        network : networkx.DiGraph
        dynamic_demand : dyntapy.demand.DynamicDemand
        simulation_time: dyntapy.time.SimulationTime
        """
        _check_centroid_connectivity(network)
        self.network = network
        self.dynamic_demand = dynamic_demand
        self.time = simulation_time
        self.internal_network = build_network(network)
        log("network build")

        self.internal_dynamic_demand: InternalDynamicDemand = (
            build_internal_dynamic_demand(
                dynamic_demand, simulation_time, self.internal_network
            )
        )
        log("demand simulation build")

    def run(self, method: str = "i_ltm_aon"):
        """

        Parameters
        ----------

        method: {'i_ltm_aon','incremental_assignment', 'aon'}

        Returns
        -------

        dyntapy.results.DynamicResult

        Notes
        -----

        All the presented options utilize the same dynamic network loading (DNL) and
        route choice component, that is the iterative link transmission model [1]_
        and an
        iterative procedure to update a time-dependent arrival map, see [2]_.

        'i_ltm_aon' refers to a dynamic deterministic user equilibrium solution,
        note that for congested networks this is not guaranteed to converge below a
        gap of 0.01.

        'incremental_assignment' assigns the demands in chunks and updates the
        both costs and route choice after each DNL.

        'aon' simply assigns all demand at free flow costs and executes one DNL.

        References
        ----------

        .. [1] Himpe, Willem, Ruben Corthout, and MJ Chris Tampère.
            ‘An Efficient Iterative Link Transmission Model’. Transportation Research
            Part B: Methodological 92 (2016): 170–90.

        .. [2] Himpe, Willem. ‘INTEGRATED ALGORITHMS FOR REPEATED DYNAMIC
            TRAFFIC ASSIGNMENTS’. KU Leuven, 2016.




        """
        dyntapy._context.running_assignment = (
            self  # making the current assignment available as global var
            # mainly for debugging
            # can be imported and used for visualization of interim computational
            # states
        )
        # TODO: add option to return iterations
        methods = {
            "i_ltm_aon": i_ltm_aon,
            "incremental_assignment": incremental,
            "aon": aon,
        }
        if method in methods:
            result = methods[method](
                self.internal_network, self.internal_dynamic_demand, self.time
            )

        else:
            raise NotImplementedError(f"{method=} is not defined ")
        return DynamicResult(**result)


class StaticAssignment:
    """

    This class stores all the information needed for the assignment itself.
    Upon initialisation both the network and demand are transformed into
    internal representations.

    Parameters
    ----------

    g : networkx.DiGraph
        road network graph with attributes and labelling as specified in the module
        description.
    od_graph : networkx.DiGraph
        graph with centroids as nodes with specified coordinates as 'x_coord' and
        'y_coord'. For each OD pair with a non-zero demand there
        is a link with a corresponding 'flow' element.


    """

    def __init__(self, g, od_graph):
        self.internal_network = build_network(g)
        log("network build")
        self.network = g
        self.od_graph = od_graph
        self.internal_demand = build_internal_static_demand(od_graph)
        self.result = None
        self.iterations = None
        log("Assignment object initialized!")
        print("init passed successfully")

    def run(self, method, store_iterations=False):
        """

        Parameters
        ----------

        method : {'dial_b','frank_wolfe', 'msa', 'sun'}
        store_iterations : bool
            set to True to get information on the individual iterations

        Returns
        -------

        dyntapy.results.StaticResult
        list of dyntapy.results.StaticResult, optional
            intermediate computation states

        Notes
        -----

        'msa', 'frank_wolfe' and 'dial_b' all try to find the static deterministic
        user equilibrium.

        'msa' refers to the Method of Successive Averages, a well known method in
        Traffic Assignments that tends to zig-zag around equilibrium.

        'frank_wolfe' refers to the Frank-Wolfe Algorithm.

        'dial_b' refers to Dial's Algorithm B, a bush-based assignment
        algorithm. It alleviates the rather slow convergence of the Frank-Wolfe
        algorithm close to equilibrium, see [3]_.

        'sun' returns a stochastic uncongested assignment of flows on the free-flow
        travel times that are determined by the lengths and speeds of the links. It
        is based on Dial's method, see [4]_. It does not consider the whole path set
        and rests the definition of 'efficient links' to allow for computations on an
        acyclic graph.

        'sun', 'frank_wolfe' and 'msa' are included for educational use.

        'dial_b' has been optimized and converges quickly even for
        large networks with thousands of links.

        References
        ----------

        .. [3] Dial, Robert B. ‘A Path-Based User-Equilibrium Traffic Assignment
            Algorithm That Obviates Path Storage and Enumeration’.
            Transportation Research Part B: Methodological 40,
            no. 10 (December 2006): 917–36. https://doi.org/10.1016/j.trb.2006.02.008.

        .. [4] Dial, Robert B. "A probabilistic multipath traffic assignment model
            which obviates path enumeration."
            Transportation research 5, no. 2 (1971): 83-111.

        """

        dyntapy._context.running_assignment = (
            self  # making the current assignment available as global var
        )
        # assignment needs to return at least link_cost and flows, ideally also
        # multi-commodity (origin, destination or origin-destination)
        if method == "dial_b":
            costs, origin_flows, gap_definition, gap = dial_b(
                self.internal_network, self.internal_demand, store_iterations
            )
            flows = np.sum(origin_flows, axis=0)
            result = StaticResult(
                costs,
                flows,
                self.internal_demand.origins,
                self.internal_demand.destinations,
                skim=get_skim(costs, self.internal_demand, self.internal_network),
                gap_definition=gap_definition,
                gap=gap,
                origin_flows=origin_flows,
            )
        elif method == "msa":
            costs, flows, gap_definition, gap = msa_flow_averaging(
                self.internal_network, self.internal_demand, store_iterations
            )
            result = StaticResult(
                costs,
                flows,
                self.internal_demand.origins,
                self.internal_demand.destinations,
                skim=get_skim(costs, self.internal_demand, self.internal_network),
                gap_definition=gap_definition,
                gap=gap,
            )
        elif method == "sun":
            assert not store_iterations
            # no iterations in uncongested assignments
            costs, flows, origin_flows = sun(
                self.internal_network, self.internal_demand
            )
            result = StaticResult(
                costs,
                flows,
                self.internal_demand.origins,
                self.internal_demand.destinations,
                skim=get_skim(costs, self.internal_demand, self.internal_network),
                origin_flows=origin_flows,
            )

        else:
            raise NotImplementedError(f"{method=} is not defined ")
        if not store_iterations:
            return result
        else:
            return result, dyntapy._context.iteration_states
