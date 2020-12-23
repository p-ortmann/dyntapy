#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import ILTMNetwork, DynamicDemand, SimulationTime, ILTMResults, StaticDemand
from dtapy.core.network_loading.i_ltm_setup import i_ltm_setup
from dtapy.parameters import i_ltm_gap,i_ltm_max_it
import numpy as np


def i_ltm(network: ILTMNetwork, dynamic_demand: DynamicDemand, results: ILTMResults, time: SimulationTime):
    all_destinations = dynamic_demand.all_destinations
    all_origins = dynamic_demand.all_origins
    tot_time_steps = time.tot_time_steps
    step_size = time.step_size

    tot_links = network.tot_links
    tot_nodes = network.tot_nodes
    tot_destinations = len(all_destinations)
    max_out_links =np.max(network.nodes.tot_out_links)
    max_in_links = np.max(network.nodes.tot_in_links)


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

    # local rename results properties
    # check for warm and cold starting is done in i_ltm_setup.py


    nodes_2_update = results.nodes_2_update
    cvn_up, cvn_down = results.cvn_up, results.cvn_down
    con_up, con_down = results.con_up, results.con_down
    marg_comp = results.marg_comp
    # allocate memory to local variables
    rf_down_cvn_db = np.zeros(tot_links)
    sf_down_cvn_db = np.zeros(tot_links, tot_destinations)


    # % allocate
    # memory
    # to
    # all
    # local
    # variables
    # RF_down_cvn_db = zeros(totLinks, 1);
    # SF_up_cvn_db = zeros(totLinks, totDestinations);
    #
    # tot_sendingFlow = zeros(maxIncomingLinks, 1);
    # temp_capacities_in = zeros(maxIncomingLinks, 1);
    # temp_capacities_out = zeros(maxOutgoingLinks, 1);
    # receivingFlow = zeros(maxOutgoingLinks, 1);
    # sendingFlow = zeros(maxIncomingLinks, totDestinations);
    temp_sending_flow = np.empty(max_in_links, tot_destinations);
    temp_receiving_flow = np.emtpy(max_out_links, tot_destinations);
    # outgoingFlow = zeros(maxOutgoingLinks, totDestinations);
    #
    # origins_t = union(origins, find(nin == 0)
    # ');
    # destinations_t = union(destinations, find(nout == 0)
    # ');
    #
    # deltaChange = zeros(totNodes, 1);
    # sortedNodes = zeros(totNodes, 1);
    #
    # turningFractions = zeros(maxIncomingLinks, maxOutgoingLinks);
    # turningFlows = zeros(maxIncomingLinks, maxOutgoingLinks);
    # forward implicit scheme
    #go sequentially over each time step
    mean_it_iltm = 0
    max_it_iltm = 0
    tot_nodes_updates = 0


# part of these assignments may be added eventually, let's see what we actually need with our TF notation
    for t in range(tot_time_steps):
        if not nodes_2_update[:,t].any():
            continue
        rf_down_cvn_bool = np.full(tot_links, True)
        rf_up_cvn_bool= np.full(tot_links, True)
        if dynamic_demand.is_loading: #check if any origins are sending flow into the network this time step
            current_demand:StaticDemand=dynamic_demand.next
            for origin in current_demand.origins:
                if nodes_2_update[origin,t]:
                    tot_nodes_updates+=1
                    out_links = network.nodes.out_links.get_nnz(origin)
                    for link in np.enumout_links:





