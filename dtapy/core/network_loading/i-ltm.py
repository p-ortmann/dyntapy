#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import ILTMNetwork, DemandSimulation, SimulationTime
from dtapy.core.network_loading.i_ltm_setup import i_ltm_setup


def i_ltm(network: ILTMNetwork, demand: DemandSimulation, results, time: SimulationTime):
    all_destinations = demand.all_destinations
    all_origins = demand.all_origins
    tot_time = time.tot_time_steps
    step_size = time.step_size

    tot_links = network.tot_links
    tot_nodes = network.tot_nodes
    tot_destinations = len(all_destinations)

    # local rename of link properties, sticking with matlab conventions here

    str_n = network.links.from_node
    end_n = network.links.to_node
    cap = network.links.capacity
    kjm = network.links.kjam
    length = network.links.length
    vind = network.links.vf_ratio
    vrt = network.links.vf_ratio
    wind = network.links.vw_index
    wrt = network.links.vw_ratio

    # local rename node properties
    in_l_csr = network.nodes.in_turns
    out_l_csr = network.nodes.out_turns
    n_in = network.nodes.tot_out_links
    n_out = network.nodes.tot_in_links

    # number of in and outcoming links and iterators can be retrieved through
