#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from dyntapy.datastructures.csr import UI32CSRMatrix
import numpy as np
from dyntapy.settings import dynamic_parameters
from warnings import warn
from numba import njit, prange
from dyntapy.dta.core.supply import Network
from dyntapy.settings import debugging

rc_precision = dynamic_parameters.route_choice.precision
dnl_precision =  dynamic_parameters.network_loading.precision

@njit()
def verify_assignment_state(network: Network, turning_fractions: np.ndarray, cvn_up: np.ndarray, cvn_down: np.ndarray,
                            tot_centroids: int):
    """
    runs all available consistency tests on the assignment state
    """
    if not debugging:
        return None
    else:
        sum_of_turning_fractions(turning_fractions, network.links.out_turns, network.links.link_type,
                                 network.turns.to_node, tot_centroids=tot_centroids)
        continuity(cvn_up, cvn_down, network.nodes.in_links,
                   network.nodes.out_links, tot_centroids=tot_centroids)
        monotonicity(cvn_up, cvn_down)
        try:
            storage(cvn_up, cvn_down, network.links.k_jam, network.links.length)
        except Exception:
            print('storage test cannot be run, missing attributes')
            pass


@njit(cache=True)
def sum_of_turning_fractions(turning_fractions: np.ndarray, out_turns: UI32CSRMatrix, link_types: np.ndarray,
                             turn_to_nodes,
                             tot_centroids: int = 0, precision: float = rc_precision):
    """
    verifies if for each link the sum of the turning
    fractions for all outgoing turns is equal to 1.
    Parameters
    ----------
    link_types: type of the links, source and sink connectors (1,-1) are excluded
    turning_fractions : array, tot_active_destinations x tot_time_steps x tot_turns
    out_turns : CSR, link x link
    turn_to_nodes : to_node for each turn
    tot_centroids :
    precision : float

    Returns
    -------

    """
    if not debugging:
        return None
    else:

        try:
            for t in prange(turning_fractions.shape[1]):
                for dest_id in range(turning_fractions.shape[0]):
                    for link in out_turns.get_nnz_rows():
                        tf_sum = 0.0
                        any_network_turns = False
                        for turn in out_turns.get_nnz(link):
                            tf_sum += turning_fractions[dest_id, t, turn]
                            if turn_to_nodes[turn] > tot_centroids:
                                any_network_turns = True
                        if np.abs(tf_sum - 1.0) > precision and len(out_turns.get_nnz(link)) != 0 and any_network_turns:
                            print("turning fraction sum violation for link " + str(link) +

                                  " at time " + str(t) + " for destination id " + str(dest_id)+ " violation: " + str(np.abs(tf_sum-1)))
                            raise ValueError
        except Exception:
            warn('sum_of_turning_fractions test failed')
            return None

        print('turning fraction sum test passed successfully')


nl_precision = dynamic_parameters.network_loading.precision


@njit(cache=True)
def continuity(cvn_up: np.ndarray, cvn_down: np.ndarray, in_links: UI32CSRMatrix,
               out_links: UI32CSRMatrix, max_delta: float = nl_precision, tot_centroids=0):
    """
    verifies for each node, destination and time step whether the sum of all
    downstream cumulatives of the incoming links equals the sum of the upstream cumulatives of all outgoing links
    Parameters
    ----------
    tot_centroids : number of centroids, assumed to be labelled as the first nodes
    cvn_up : upstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    cvn_down : downstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    in_links : CSR node x links
    out_links : CSR node x links
    max_delta : float, allowed constraint violation

    Returns
    -------
    """
    if not debugging:
        return None
    else:

        tot_time_steps = cvn_down.shape[0]
        tot_destinations = cvn_down.shape[2]
        try:
            for t in prange(tot_time_steps):
                for d in prange(tot_destinations):
                    for node in range(tot_centroids, in_links.nnz_rows.size):
                        in_flow = 0.0
                        out_flow = 0.0
                        for in_link in in_links.get_nnz(node):
                            in_flow += cvn_down[t, in_link, d]
                        for out_link in out_links.get_nnz(node):
                            out_flow += cvn_up[t, out_link, d]
                        if np.abs(out_flow - in_flow) > max_delta:
                            print("continuity violation in node " + str(node) +
                                  " at time " + str(t) + " for destination id " + str(d)+' outflow - inflow: ' + str(out_flow - in_flow))
                            raise ValueError
        except Exception:
            warn('continuity test failed')
            return None
        print('continuity test passed successfully')


@njit(cache=True)
def monotonicity(cvn_up, cvn_down):
    """

    Parameters
    ----------
    cvn_up : upstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    cvn_down : downstream cumulative numbers, tot_time_steps x tot_links x tot_destinations

    Returns
    -------

    """
    if not debugging:
        return None
    else:
        tot_time_steps = cvn_down.shape[0]
        tot_links = cvn_down.shape[1]
        tot_destinations = cvn_down.shape[2]
        try:
            for t in prange(tot_time_steps - 1):
                for link in prange(tot_links):
                    for d in range(tot_destinations):
                        if cvn_up[t, link, d] > cvn_up[t + 1, link, d] or cvn_down[t, link, d] > cvn_down[t + 1, link, d]:
                            print("monotonicity violation for link " + str(link) +
                                  " at time " + str(t) + " for destination id " + str(d))
                            raise ValueError
        except Exception:
            warn('monotonicity test failed')
            return None
        print('monotonicity test passed successfully')

@njit
def storage(cvn_up: np.ndarray, cvn_down: np.ndarray, jam_density: np.ndarray, length: np.ndarray):
    """

    Parameters
    ----------
    cvn_up : upstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    cvn_down : downstream cumulative numbers, tot_time_steps x tot_links x tot_destinations
    jam_density : max density for each link in veh/km
    length : for each link in km

    Returns
    -------

    """
    if not debugging:
        return None
    else:
        tot_time_steps = cvn_down.shape[0]
        tot_links = cvn_down.shape[1]
        try:
            for t in prange(tot_time_steps - 1):
                for link in prange(tot_links):
                    if np.sum(cvn_up[t, link, :] - cvn_down[t, link, :]) > jam_density[link] * length[link]+dnl_precision:
                        print("storage violation for link " + str(link) +
                              " at time " + str(t))
                        raise ValueError
                    if np.sum(cvn_up[t, link, :] - cvn_down[t, link, :])<-dnl_precision:
                        print('negative queue length for link ' + str(link)+' at time '+str(t))
                        raise ValueError

        except Exception:
            warn('storage test failed')
            return None
        print('storage test passed successfully')