import numpy as np

from dyntapy import StaticAssignment, StaticResult, show_network
from dyntapy.demand import InternalStaticDemand
from dyntapy.supply import Network


def loading(
    demand: InternalStaticDemand,
    network: Network,
    flows,
    numerical_threshold=0.01,  # demand object is float32, precision is limited to
    # approx. 6 digits (np.finfo(np.float32).precicion)
):
    tot_centroids = network.nodes.is_centroid.sum()
    origin_violations = np.full(tot_centroids, False)
    destination_violations = np.full(tot_centroids, False)
    to_destinations = demand.to_destinations
    to_origins = demand.to_origins
    origin_links = demand.origins
    destination_links = np.empty_like(demand.destinations)
    for idx, destination in enumerate(demand.destinations):
        destination_links[idx] = network.nodes.in_links.get_nnz(destination)[0]
    for origin in origin_links:
        origin_violations[origin] = (
            np.abs(flows[origin] - to_destinations.get_row(origin).sum())
            > numerical_threshold
        )

        if origin_violations[origin]:
            print(f"violation for {origin=};")
            print(
                f"magnitude: "
                f"{np.abs(flows[origin] - to_destinations.get_row(origin).sum())}"
            )
    for link, destination in zip(destination_links, demand.destinations):
        destination_violations[destination] = (
            np.abs(flows[link] - to_origins.get_row(destination).sum())
            > numerical_threshold
        )
        if destination_violations[destination]:
            print(f"violation for {destination=};")
            print(
                f"magnitude: "
                f"{np.abs(flows[link] - to_origins.get_row(destination).sum())}"
            )

    return (
        np.any(origin_violations) or np.any(destination_violations),
        origin_violations,
        destination_violations,
    )


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
                print(f"continuity_violation for {node=}")
                print(f"{out_flow-in_flow=}")
                continuity_violations[node] = True
                values[node] = out_flow - in_flow
                for link in network.nodes.out_links.get_nnz(node):
                    links_to_highlight.append(link)
                for link in network.nodes.in_links.get_nnz(node):
                    links_to_highlight.append(link)

    return continuity_violations, values, links_to_highlight


def debug_assignment_result(result: StaticResult, assignment: StaticAssignment):
    demand = assignment.internal_demand
    network = assignment.internal_network
    flows = result.flows
    loading_violation, origin_violations, destination_violations = loading(
        demand, network, flows
    )
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
