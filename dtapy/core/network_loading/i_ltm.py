#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
from dtapy.core.jitclasses import ILTMNetwork, DynamicDemand, SimulationTime, ILTMResults, StaticDemand
import numpy as np
from dtapy.parameters import LTM_GAP as gap
from numba import njit
import warnings

@njit
def i_ltm(network: ILTMNetwork, dynamic_demand: DynamicDemand, results: ILTMResults, time: SimulationTime):
    all_destinations = dynamic_demand.all_destinations
    all_origins = dynamic_demand.all_origins
    tot_time_steps = time.tot_time_steps
    step_size = time.step_size

    tot_links = network.tot_links
    tot_nodes = network.tot_nodes
    tot_destinations = dynamic_demand.tot_destinations
    max_out_links = np.max(network.nodes.tot_out_links)
    max_in_links = np.max(network.nodes.tot_in_links)

    # local rename of link properties, sticking with matlab conventions here

    # str_n = network.links.from_node
    # end_n = network.links.to_node
    to_node = network.links.to_node
    from_node = network.links.from_node
    cap = network.links.capacity
    kjm = network.links.k_jam
    length = network.links.length
    vind = network.links.vf_index
    vrt = network.links.vf_ratio
    wind = network.links.vw_index
    wrt = network.links.vw_ratio

    # local rename node properties
    in_l_csr = network.nodes.in_links
    out_l_csr = network.nodes.out_links
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
    sf_down_cvn_db = np.zeros((tot_destinations, tot_links))

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
    temp_sending_flow = np.empty((max_in_links,tot_destinations))
    temp_receiving_flow = np.empty((max_out_links, tot_destinations))
    # deviating from Matlab here with 'C' order, access happens usually per link and for all destinations
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
    # go sequentially over each time step
    mean_it_iltm = 0
    max_it_iltm = 0
    tot_nodes_updates = 0

    # part of these assignments may be added eventually, let's see what we actually need with our TF notation

    for t in range(1, tot_time_steps):
        if not nodes_2_update[:, t].any():
            continue
        rf_down_cvn_bool = np.full(tot_links, True)
        rf_up_cvn_bool = np.full(tot_links, True)
        if dynamic_demand.is_loading:  # check if any origins are sending flow into the network this time step
            current_demand: StaticDemand = dynamic_demand.next
            for origin in current_demand.origins:
                if nodes_2_update[t, origin]:
                    tot_nodes_updates += 1
                    out_links = network.nodes.out_links.get_nnz(origin)
                    for index, link in np.ndenumerate(out_links):
                        temp_sending_flow[0, : ] = cvn_up[ t - 1, link,:]
                        for dest, val in zip(current_demand.to_destinations.get_nnz(origin),
                                             current_demand.to_destinations.get_row(origin)):
                            temp_sending_flow[ 0, dest] += val

                        if np.sum(np.abs(temp_sending_flow[0, :] - cvn_up[ t - 1, link, :])) > gap:
                            nodes_2_update[ min(tot_time_steps, t + 1), origin] = True
                            cvn_up[t, link, :] = temp_sending_flow[:, 0]
                            if np.sum(cvn_up[t, link, :] - cvn_up[t-1, link, :]) < cap[link] * step_size:
                                con_up[t, out_links] = False
                            else:
                                con_up[t, out_links] = True

                        if vind[link] == -1:
                            nodes_2_update[t, to_node[link]] = True
                        else:
                            try:
                                nodes_2_update[ t - vind[link] - 1, to_node[link]] = True
                                nodes_2_update[ t - vind[link], to_node[link]] = True
                            except Exception:
                                assert t - vind[link] > tot_time_steps
                                if t - vind[link] - 1 == tot_time_steps:
                                    nodes_2_update[ tot_time_steps, to_node[link]] = True
                                else:
                                    print('Simulation time period is too short for given demand.'
                                              ' Not all vehicles are able to exit the network')
    print('origin loading sucessful')
    print('boooooooo')
