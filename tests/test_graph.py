import networkx as nx
import pytest

from tests.helpers import make_graph


def test_nodes_are_exposed_as_strings(ring_graph):
    assert ring_graph.nodes == ["0", "1", "2", "3"]


def test_neighbors_on_ring(ring_graph):
    assert sorted(ring_graph.neighbors("0")) == ["1", "3"]
    assert sorted(ring_graph.neighbors(2)) == ["1", "3"]  # int ids work too


def test_synthetic_graph_gets_unit_weights():
    g = make_graph(nx_class="complete_graph", n=3)

    assert len(g.nodes) == 3
    assert g.get_weight("0", "1") == 1


def test_missing_source_raises():
    with pytest.raises(ValueError):
        make_graph()


def test_star_hub_is_connected_to_everyone():
    g = make_graph(graph=nx.star_graph(4))  # hub 0 + 4 leaves

    assert sorted(g.neighbors("0")) == ["1", "2", "3", "4"]
    assert g.neighbors("3") == ["0"]
