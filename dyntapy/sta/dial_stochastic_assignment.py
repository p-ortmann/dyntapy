from numba import njit, prange
from dyntapy.sta.utilities import (
    generate_bushes_line_graph,
    __link_to_turn_cost_static,
    __get_u_turn_turn_restrictions,
    __bpr_cost,
)
from dyntapy.graph_utils import _make_out_links, _make_in_links
from dyntapy.settings import parameters
from dyntapy.visualization import show_network
import numpy as np


@njit
def _dial_network_loading(
        topological_orders, turning_fractions, out_turns, demand, node_destinations
):
    tot_links = topological_orders.shape[1]
    destination_flows = np.zeros((demand.destinations.size, tot_links))

    for d_id in prange(node_destinations.size):
        node_destination = node_destinations[d_id]

        for j in topological_orders[d_id][::-1]:
            if j in demand.origins:
                if node_destination in demand.to_destinations.get_nnz(j):
                    _id = np.argwhere(
                        demand.to_destinations.get_nnz(j) == node_destination
                    ).flatten()[0]
                    destination_flows[d_id, j] = (
                            destination_flows[d_id, j]
                            + demand.to_destinations.get_row(j)[_id]
                    )

            for out_link, turn in zip(out_turns.get_row(j), out_turns.get_nnz(j)):
                destination_flows[d_id, out_link] = (
                        destination_flows[d_id, out_link]
                        + turning_fractions[d_id, turn] * destination_flows[d_id, j]
                )

    # hence the workaround below

    return destination_flows, destination_flows.sum(axis=0)


@njit()
def _set_labels(
        destination,
        out_turns,
        in_turns,
        turns_in_bush,
        distances,
        topological_order,
        from_links,
        to_links,
        turn_costs,
        mu,
):
    # the distances here are kept steady between iterations, they just resolve a
    # numerical issue.
    # only the turn costs vary when using this for iterative schemes.

    turn_utility = np.zeros(turn_costs.size, dtype=np.float64)
    link_weights = np.zeros(distances.size, dtype=np.float64)
    link_weights[destination] = 1.0
    turn_weights = np.zeros(turn_costs.size, dtype=np.float64)
    if np.exp(-turn_costs.max() * 1 / mu) == 0:
        raise ValueError('mu too small for the max cost in the given network')
    for turn, (in_bush, i, j) in enumerate(zip(turns_in_bush, from_links, to_links)):
        # larger theta leads to AON behavior
        if in_bush:
            turn_utility[turn] = np.exp(1 / mu * (-turn_costs[turn]))
    for j in topological_order:
        if j != destination:
            link_weights[j] = 0.0
            for i, turn in out_turns[j]:
                link_weights[j] += turn_weights[turn]
            # assert node_weights[i] > 0.00001
        for i, turn in in_turns[j]:
            turn_weights[turn] = turn_utility[turn] * link_weights[j]
    return turn_weights, link_weights


@njit
def _get_tf(
        tot_links,
        from_links,
        to_links,
        link_destinations,
        tot_turns,
        turns_in_bush,
        distances,
        turn_costs,
        topological_orders,
        mu,
        all_bush_in_turns,
        all_bush_out_turns
):
    tot_destinations = link_destinations.size
    turning_fractions = np.zeros((tot_destinations, tot_turns))
    turn_weights = np.zeros((tot_destinations, tot_turns))
    link_weights = np.zeros((tot_destinations, tot_links))
    for d_id in prange(link_destinations.size):
        destination = link_destinations[d_id]
        bush_out_turns = all_bush_out_turns[d_id]
        bush_in_turns = all_bush_in_turns[d_id]

        turn_weights[d_id], link_weights[d_id] = _set_labels(
            destination,
            bush_out_turns,
            bush_in_turns,
            turns_in_bush[d_id],
            distances[d_id],
            topological_orders[d_id],
            from_links,
            to_links,
            turn_costs,
            mu,
        )
        for j in topological_orders[d_id]:
            if link_weights[d_id][j] == 0:
                continue  # no path from this link, or all existing paths have a
                # prohibitively large cost
            for out_link, turn in bush_out_turns[j]:
                turning_fractions[d_id, turn] = turn_weights[d_id, turn] / \
                                                link_weights[d_id][j]

    return link_weights, turn_weights, turning_fractions


@njit
def _dial_sue(network, demand, topo_costs, mu, max_it, max_gap):
    destination_links = np.empty_like(demand.destinations)
    for _id, dest in enumerate(demand.destinations):
        destination_links[_id] = network.nodes.in_links.get_nnz(dest)[0]
    tot_turns = network.tot_turns
    tot_links = network.tot_links
    turns = network.turns
    links = network.links
    link_ff_tt = links.length / links.free_speed

    turn_restr = __get_u_turn_turn_restrictions(
        tot_turns, turns.from_node, turns.to_node
    )
    turn_costs = __link_to_turn_cost_static(
        tot_turns, turns.from_link, topo_costs, turn_restr
    )
    topological_orders, turns_in_bush, topo_distances, all_bush_in_turns, \
    all_bush_out_turns = \
        generate_bushes_line_graph(
            turn_costs,
            turns.from_link,
            turns.to_link,
            links.in_turns,
            destination_links,
            tot_links,
        )

    # initial state with no flows in the network and free flow travel times
    c2 = np.copy(link_ff_tt).astype(np.float64)
    f1 = np.zeros(tot_links, dtype=np.float64)
    gap = np.inf
    k = 0
    while gap > max_gap and k < max_it + 1:
        k = k + 1
        turn_costs = __link_to_turn_cost_static(
            tot_turns, turns.from_link, c2, turn_restr
        )
        # topo costs remain unchanged, included for numerical reasons, explicitly NOT
        # part of the iterative scheme.

        _, _, turning_fractions = _get_tf(
            tot_links,
            turns.from_link,
            turns.to_link,
            destination_links,
            tot_turns,
            turns_in_bush,
            topo_distances,
            turn_costs,
            topological_orders,
            mu,
            all_bush_in_turns,
            all_bush_out_turns
        )
        destination_flows, f2 = _dial_network_loading(
            topological_orders,
            turning_fractions,
            links.out_turns,
            demand,
            demand.destinations,
        )
        c1 = np.copy(c2)
        c2 = __bpr_cost(f2, links.capacity, link_ff_tt)
        # f2 = 1 / k * f2 + (k - 1) / k * f1
        # print(f'iteration k ={float(k)} and gap = {float(gap)}')
        if k > 1:
            f2 = 0.5 * f2 + 0.5 * f1
            gap = np.nanmax(np.abs(f2 - f1) / f1)  # value warning for true_divide is
            # normal here
            # from dyntapy._context import running_assignment
            # show_network(running_assignment.network, flows=f2, link_kwargs={
            #     'cur_costs': c1, 'new_costs': c2}, title=f'iteration_{k=}_and_{gap=}')
        # print(f'finished it {k}')
        f1 = np.copy(f2)
    if k == max_it:
        print("max iterations reached")
    return c1, f2, destination_flows


def dial_sue(
        network,
        demand,
        link_costs=None,
        mu=parameters.static_assignment.mu,
        max_iterations=parameters.static_assignment.sue_dial_max_iterations,
        max_gap=parameters.static_assignment.sue_dial_gap,
):
    """
    A stochastic static traffic assignment routine using Dial's Algorithm relying on
    topological orders.

    Parameters
    ----------
    network : dyntapy.Network
    demand : dyntapy.InternalStaticDemand
    link_costs : np.ndarray, optional
        costs to generate the topological order(s) from, if not provided free flow
        travel times will be used.
    mu: float, optional
        scaling parameter for logit, correlated with the standard deviation.
        Deterministic  behaviours when equal to 0, recommended range: [0,1]
    max_iterations: int, optional
        maximum number of iterations after which to terminate the procedure
        independent of the precision reached.
    max_gap: float, optional
        allowed gap for the flows to be considered in equilibrium.

    Notes
    ------

    The scaling parameter mu correlates with the standard deviation.

    Returns
    -------

    """
    if link_costs is None:
        link_costs = network.links.length / network.links.free_speed
    costs, flows, destination_flows = _dial_sue(
        network, demand, link_costs, mu, max_iterations, max_gap
    )
    return costs, flows, destination_flows
