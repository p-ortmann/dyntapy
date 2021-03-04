import numpy as np
from numba.typed import List
from numba.core.types.containers import ListType
from numba import int8, njit


def orca_node_model(sending_flow, turning_fractions, turning_flows, receiving_flow,
                    turn_capacity, in_link_capacity):
    """

    Parameters
    ----------
    sending_flow : array, 1D
    turning_fractions : array, dim tot_in_links x tot_out_links
    turning_flows : array, dim tot_in_links x tot_out_links
    receiving_flow : array, 1D
    turn_capacity : array, 1D
    in_link_capacity : array, 1D

    Returns
    -------
    turn flows according to oriented capacity (ORCA) principles, not taking into account signals
    see Tampère, Chris MJ, et al. "A generic class of first order node models for
    dynamic macroscopic simulation of traffic flows." Transportation Research Part B:
    Methodological 45.1 (2011): 289-309 for further details and background
    The variable and function names chosen here are aligned with what's shown in the paper

    """
    (tot_in_links, tot_out_links) = turning_fractions.shape
    # changing names of variables so that they correspond to notation cited above
    R = np.copy(receiving_flow)
    S = turning_flows  # initialized externally
    s = sending_flow  # differs from paper
    c = in_link_capacity  # differs from paper
    C = _calc_oriented_capacities(turning_fractions, in_link_capacity,
                                  turn_capacity.reshape(turning_fractions.shape), use_turn_cap=True)
    q = np.zeros((tot_in_links, tot_out_links), dtype=np.float32)
    # J being the set of out_links with demand towards them
    J = List(np.where(np.sum(turning_fractions, 0) > 0)[0])
    # U is a list of lists with U[j] being the current contenders (in_links i) of out_link j
    U = List()
    j_bucket = List(lsttype=ListType(int8))
    i_bucket = List(lsttype=ListType(int8))
    for j in range(tot_out_links):
        U.append(List(np.where(turning_fractions[:, j] > 0)[0]))
    a = np.full(tot_out_links, np.inf, dtype=np.float32)  # init of a with fixed size
    while len(J) > 0:
        a, min_a, _j = __find_most_restrictive_constraint(J, R, U, C, a)
        # print(f'new {_j}')
        __impose_constraint(_j, min_a, a, U, c, s, S, q, J, R, C, i_bucket, j_bucket)

    return q


def __impose_constraint(_j, min_a, a, U, c, s, S, q, J, R, C, i_bucket, j_bucket):
    # loosely corresponds to step 4, pg 301
    all_in_links_supply_constrained = True
    for i in U[_j]:
        if s[i] <= min_a * c[i]:  # if in_link i is not supply constrained
            all_in_links_supply_constrained = False
            for j in J:
                # it can send fully to all its out_links
                if S[i][j] > 0:  # if in_link i is competing for out_link j
                    q[i, j] = S[i][j]
                    # to verify: does this implicitly respect turn capacities if the
                    # S[i][j] are set appropriately - I reckon it does.
                    # i is demand constrained, it takes its share from each supply j
                    # corresponding to the demand.
                    R[j] = R[j] - S[i][j]
                    if j == _j:
                        i_bucket.append(i)  # to remove i from U[_j] after looping
                    else:
                        U[j].remove(i)  # other sets are stable ..
                    # removing outside the loops to keep the set stable for iterations
                    if len(U[j]) == 0:
                        j_bucket.append(j)  # to remove j after looping

    while len(i_bucket) > 0:
        i = i_bucket.pop(0)
        U[_j].remove(i)
    if len(U[_j]) == 0:
        j_bucket.append(j)
    if all_in_links_supply_constrained:
        for i in U[_j]:
            # all in_links of j get capacity proportional share
            for j in J:
                q[i][j] = C[i][j] * min_a
                R[j] = R[j] - q[i][j]
                # since i is constrained by _j, it takes it oriented
                # capacity proportional share from all j
                if j != _j and q[i][j] > 0:
                    # i is constrained by _j and not by other j
                    # therefore, it drops out of all competition sets U[j]
                    U[j].remove(i)
                    if len(U[j]) == 0:
                        a[j] = np.inf
                        j_bucket.append(j)
        a[_j] = np.inf
        J.remove(_j)
    while len(j_bucket) > 0:
        j = j_bucket.pop(0)
        a[j] = np.inf
        J.remove(j)


def __find_most_restrictive_constraint(J, R, U, C, a):
    # loosely corresponds to step 3, pg 301
    for j in J:
        R_j = R[j]
        if len(U[j]) == 0:
            # no competing links for j
            # print(f'a is one here {a}')
            continue
        else:
            sum_c_ij = np.int(0)
            for i in U[j]:
                sum_c_ij += C[i, j]
            a[j] = R_j / sum_c_ij
    _j = np.argmin(a)
    return a, a[_j], _j,  # determine most restrictive out_link j


def _calc_oriented_capacities(turning_fractions, in_link_capacity, turn_capacity, use_turn_cap=False):
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
    dim = turning_fractions.shape  # (tot_in_links x tot_out_links)
    oriented_capacity = np.empty_like(turning_fractions)
    turn_capacity.reshape(dim)
    if use_turn_cap:
        for in_link, cap in zip(range(dim[0]), in_link_capacity):
            for out_link in range(dim[1]):
                oriented_capacity[in_link][out_link] = min((turn_capacity[in_link, out_link],
                                                            turning_fractions[in_link, out_link] * cap))
    else:
        for in_link, cap in zip(range(dim[0]), in_link_capacity):
            oriented_capacity[in_link] = turning_fractions[in_link] * cap
    return oriented_capacity
