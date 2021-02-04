import numpy as np
from numba import njit
from numba import List


def orca_node_model(tot_in_links, tot_out_links, sending_flow, turning_fractions, turning_flows, receiving_flow,
                    turn_capacity, in_link_capacity):
    '''

    Parameters
    ----------
    tot_in_links : int, number of in_links
    tot_out_links : int, number of out_links
    sending_flow : array, 1D
    turning_fractions : array, dim >= tot_in_links x tot_out_links
    turning_flows : array, dim >= tot_in_links x tot_out_links
    receiving_flow : array, 1D
    turn_capacity : array, 1D
    in_link_capacity : array, 1D

    Returns
    -------
    turn flows according to oriented capacity (ORCA) principles, not taking into account signals
    see Tampère, Chris MJ, et al. "A generic class of first order node models for
    dynamic macroscopic simulation of traffic flows." Transportation Research Part B:
    Methodological 45.1 (2011): 289-309 for further details and background
    The variable names chosen here are aligned with what's shown in the paper
    '''

    R = np.copy(receiving_flow)
    competing_turns = turning_flows.flatten()
    competing_turns[competing_turns > 0] = 1
    # J being the set of out_links with demand towards them
    J = np.where(np.sum(turning_fractions, 0) > 0)[0]
    # U is a list of lists with U[j] being the current contenders (in_links i) of out_link j
    U = List()
    a = np.full(tot_out_links, 100000)  # init of a with fixed size

    for j in range(tot_out_links):
        U[j] = List(np.where(turning_fractions[:, j] > 0)[0])
    oriented_capacity = _calc_oriented_capacities(turning_fractions, in_link_capacity,
                                                  turn_capacity.reshape(turning_fractions.shape), use_turn_cap=True)
    while len(J) > 0:
        _j = __determine_most_restrictive_constraint(R, U, oriented_capacity, a)


def __determine_most_restrictive_constraint(R, U, oriented_capacity, a):
    for j, R_j in enumerate(R):
        if U[j].size == 0:
            # no competing links for j
            continue
        else:
            sum_c_ij = np.int(0)
            for i in U[j]:
                sum_c_ij += oriented_capacity[i, j]
            a[j] = R_j / sum_c_ij
    return np.argmin(a)  # determine most restrictive out_link j


@njit
def _calc_oriented_capacities(tot_in_links, turning_fractions, in_link_capacity, turn_capacity, use_turn_cap=False):
    """

    Parameters
    ----------
    turning_fractions : array, in_links x out_links
    in_link_capacity : array, 1D
    turn_capacity : array, in_links x out_links
    use_turn_cap : bool, consider turn capacities as constraining or not

    Returns
    -------

    """
    oriented_capacity = np.empty_like(turning_fractions)
    turn_capacity.reshape(turning_fractions.shape)
    if use_turn_cap:
        for in_link, cap in zip(range(tot_in_links), in_link_capacity):
            oriented_capacity[in_link] = np.min((turn_capacity[in_link], turning_fractions[in_link] * cap))
    else:
        for in_link, cap in zip(range(tot_in_links), in_link_capacity):
            oriented_capacity[in_link] = turning_fractions[in_link] * cap
    return oriented_capacity
