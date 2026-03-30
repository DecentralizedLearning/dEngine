from dengine.graph import Graph


def _neighbor_count(network: Graph, node_id) -> int:
    neighbors = list(network.neighbors(node_id))
    return len(neighbors)


def nodes_sorted_by_neighbor_count(network: Graph):
    nodes = network.nodes
    return sorted(
        nodes,
        key=lambda _x: _neighbor_count(network, _x),
        reverse=True
    )
