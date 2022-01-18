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
import networkx as nx
from typing import List, Union, Tuple
import numpy as np

import dyntapy._context
from dyntapy.demand import (
    DynamicDemand,
    InternalDynamicDemand,
    build_static_demand,
    _build_dynamic_demand,
)
from dyntapy.demand_data import _check_centroid_connectivity
from dyntapy.dta.aon import aon
from dyntapy.dta.i_ltm_aon import i_ltm_aon
from dyntapy.dta.incremental_assignment import incremental
from dyntapy.dta.time import SimulationTime
from dyntapy.sta.dial_b import dial_b
from dyntapy.sta.msa import msa_flow_averaging
from dyntapy.sta.uncongested_dial import sun
from dyntapy.supply_data import build_network
from dyntapy.utilities import log
from dyntapy.results import StaticResult, get_skim, DynamicResult


class DynamicAssignment:
    """This class stores all the information needed for the assignment itself.
    It takes all the information from the nx.MultiDiGraph and the
    DynamicDemand and translates it into internal representations that can be
    understood by numba.
    """

    def __init__(
        self,
        network: nx.DiGraph,
        dynamic_demand: DynamicDemand,
        simulation_time: SimulationTime,
    ):
        """

        Parameters
        ----------
        network : nx.MultiDiGraph
        dynamic_demand : DynamicDemand
        """
        # the data structures starting with _ refer to internal compiled structures,
        # if you want to change them
        # you have to be familiar with numba
        _check_centroid_connectivity(network)
        self.network = network
        self.dynamic_demand = dynamic_demand
        self.time = simulation_time
        # get adjacency from nx, and
        # self.demand = self.build_demand()
        self.internal_network = build_network(network)
        log("network build")

        self.internal_dynamic_demand: InternalDynamicDemand = _build_dynamic_demand(
            dynamic_demand, simulation_time, self.internal_network
        )
        log("demand simulation build")

    def run(self, method: str = "i_ltm_aon"):
        dyntapy._context.running_assignment = (
            self  # making the current assignment available as global var
            # mainly for debugging
            # can be imported and used for visualization of interim computational
            # states
        )
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


# remapping of od from different time granularity to computation time steps
# node or link event which triggers a change in otherwise stationary characteristics
# example ramp metering event capacity choke, relative and absolute events
class StaticAssignment:
    def __init__(self, g: nx.DiGraph, od_graph: nx.DiGraph):
        """

        Parameters
        ----------
        g : nx.DiGraph
        od_matrix : array like object
            Dimensions should be nodes x nodes of the nx.DiGraph in the Assignment
            object
        """

        # TODO: make all functions use the same network definition as dynamic
        self.internal_network = build_network(g)
        log("network build")
        self.network = g
        self.od_graph = od_graph
        self.internal_demand = build_static_demand(od_graph)
        self.result = None
        self.iterations = None
        log("Assignment object initialized!")
        print("init passed successfully")

    def run(
        self, method: str = "dial_b", store_iterations: bool = False
    ) -> Union[StaticResult, Tuple[StaticResult, List[StaticResult]]]:
        """

        Parameters
        ----------
        method : ["frank_wolfe", "dial_b", "msa", "sun"]
        store_iterations :

        Returns
        -------

        """
        # TODO: run assignments in the same format as in dynamic
        dyntapy._context.running_assignment = (
            self  # making the current assignment available as global var
        )
        if method == "dial_b":
            # assignment needs to return at least link_cost and flows, ideally also
            # multi-commodity (origin, destination or origin-destination)
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
