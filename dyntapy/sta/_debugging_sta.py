from dyntapy import StaticResult, StaticAssignment, show_network
from dyntapy.supply import Network
import numpy as np


def continuity(
    flows: np.ndarray, network: Network, numerical_threshold=np.finfo(np.float32).eps
):
    continuity_violations = np.full(network.tot_nodes, False)
    values = np.zeros(continuity_violations.size)
    links_to_highlight = []
    for node in range(network.tot_nodes):
        if not network.nodes.is_centroid[node]:
            in_flow = 0
            out_flow = 0
            for link in network.nodes.in_links.get_nnz(node):
                in_flow += flows[link]
            for link in network.nodes.out_links.get_nnz(node):
                out_flow += flows[link]
            if np.abs(out_flow - in_flow) > numerical_threshold:
                continuity_violations[node] = True
                values[node] = out_flow - in_flow
                for link in network.nodes.out_links.get_nnz(node):
                    links_to_highlight.append(link)
                for link in network.nodes.in_links.get_nnz(node):
                    links_to_highlight.append(link)

    return continuity_violations, values, links_to_highlight


def debug_assignment_result(result: StaticResult, assignment: StaticAssignment):
    continuity_violations, violations = continuity(
        result.flows, assignment.internal_network
    )
    if np.any(continuity_violations):
        print("continuity violations found")
        print("plotting continuity violations")
        show_network(
            assignment.network,
            flows=result.flows,
            highlight_nodes=np.argwhere(continuity_violations).flatten(),
            node_kwargs={"continuity_violation": violations},
        )
