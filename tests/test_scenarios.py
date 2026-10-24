"""Tests for ``dengine.scenarios``: clients (how they aggregate) and engines (how they run).

Models are ``ScalarModel``s, so a model is just one number. Clients built by
``scenario_with_weights`` start with a weight equal to their node id:
on the 4-node ring, client "0" has weight 0 and neighbours "1" (1) and "3" (3).
"""
from typing import Any, Dict, cast

import networkx as nx
import pytest

from dengine.interfaces import ClientInterface
from dengine.scenarios import (
    CentralizedScenarioEngine,
    DecAvgClient,
    ScenarioPlaceholder,
    VanillaDecentralizedSequential,
    VanillaFederatedScenario,
)
from dengine.scenarios.decdiff import DecDiffClient
from dengine.scenarios.decentralized import (
    DecentralizedShallowSyncReference,
    GossipClient,
    VanillaDecentralizedMessage,
    VanillaDecentralizedSynch,
)
from dengine.scenarios.federated import FederatedClient, ServerMockClient, VanillaFederatedMessage
from dengine.scenarios.scenario import AbstractScenarioEngine
from dengine.scenarios.utils import SerialExecutor, executor_context

from tests.helpers import engine_of, make_graph, ring, scenario_inputs, set_weight, weight

DEC_AVG: Dict[str, Any] = {"use_weighted_avg": False}


def scenario_with_weights(scenario: AbstractScenarioEngine) -> AbstractScenarioEngine:
    """Give every client a weight equal to its node id, then return the scenario."""
    for uuid, client in scenario.clients.items():
        set_weight(client, float(uuid))
    return scenario


def ring_clients(client: str, arguments: Dict[str, Any] | None = None) -> Dict[str, ClientInterface]:
    """Clients of a 4-node ring, built by a scenario that does nothing on its own."""
    scenario = ScenarioPlaceholder(**scenario_inputs(ring(4), client, arguments))
    return scenario_with_weights(scenario).clients


def message_from(client: ClientInterface, time: float = 0.0) -> VanillaDecentralizedMessage:
    """A decentralized message carrying ``client``'s model."""
    return VanillaDecentralizedMessage(time=time, source_client=cast(DecAvgClient, client))


# --- Clients: receiving messages ------------------------------------------------

class TestSynchronization:
    """``AbstractClient.synchronization``: which buffered messages a client uses."""

    def test_keeps_only_the_latest_message_per_sender(self):
        clients = ring_clients("DecAvgClient", DEC_AVG)
        receiver = clients["0"]
        old, new = message_from(clients["1"], time=0), message_from(clients["1"], time=1)
        receiver.message_buffer.put(old)
        receiver.message_buffer.put(new)

        assert receiver.synchronization(current_time=1) == [new]

    def test_messages_from_the_future_wait_in_the_buffer(self):
        clients = ring_clients("DecAvgClient", DEC_AVG)
        receiver = clients["0"]
        receiver.message_buffer.put(message_from(clients["1"], time=5))

        assert receiver.synchronization(current_time=1) == []
        assert receiver.message_buffer.qsize() == 1


# --- Clients: aggregating models ------------------------------------------------

class TestDecAvgClient:
    """``DecAvgClient``: replaces its model with the average of its neighbours'."""

    def test_plain_average_of_neighbours(self):
        clients = ring_clients("DecAvgClient", DEC_AVG)

        clients["0"].aggregation([message_from(clients["1"]), message_from(clients["3"])])

        assert weight(clients["0"]) == pytest.approx(2.0)  # (1 + 3) / 2

    def test_include_myself_adds_own_model(self):
        clients = ring_clients("DecAvgClient", {**DEC_AVG, "include_myself": True})

        clients["0"].aggregation([message_from(clients["1"]), message_from(clients["3"])])

        assert weight(clients["0"]) == pytest.approx(4 / 3)  # (0 + 1 + 3) / 3

    def test_weighted_average_uses_training_set_sizes(self):
        clients = ring_clients("DecAvgClient", {"use_weighted_avg": True})
        cast(DecAvgClient, clients["1"]).train_size = 3
        cast(DecAvgClient, clients["3"]).train_size = 1

        clients["0"].aggregation([message_from(clients["1"]), message_from(clients["3"])])

        assert weight(clients["0"]) == pytest.approx(1.5)  # (3 * 1 + 1 * 3) / 4

    def test_no_messages_leaves_the_model_unchanged(self):
        clients = ring_clients("DecAvgClient", DEC_AVG)

        clients["1"].aggregation([])

        assert weight(clients["1"]) == 1.0

    def test_rejects_messages_from_non_neighbours(self):
        clients = ring_clients("DecAvgClient", DEC_AVG)

        with pytest.raises(ValueError):
            clients["0"].aggregation([message_from(clients["2"])])  # 0 and 2 are not linked


class TestDecDiffClient:
    """``DecDiffClient``: moves towards the neighbours' average by a bounded step."""

    def test_step_towards_the_average(self):
        clients = ring_clients("DecDiffClient", DEC_AVG)

        clients["0"].aggregation([message_from(clients["1"]), message_from(clients["3"])])

        # w - (w - avg) / (|w - avg| + 1) = 0 - (0 - 2) / 3
        assert weight(clients["0"]) == pytest.approx(2 / 3)

    def test_refuses_include_myself(self):
        with pytest.raises(ValueError):
            ring_clients(DecDiffClient.__name__, {**DEC_AVG, "include_myself": True})


@pytest.mark.xfail(strict=True, reason="Bug: GossipClient.aggregation never writes the new weights back to the model")
def test_gossip_client_moves_towards_neighbours():
    """``GossipClient`` with gamma=0.5 should move halfway towards its neighbour."""
    clients = ring_clients(GossipClient.__name__, {"gamma": 0.5})

    clients["0"].aggregation([message_from(clients["1"])])

    assert weight(clients["0"]) == pytest.approx(0.5)


def test_federated_client_loads_the_global_model():
    """``FederatedClient`` simply copies the model sent by the server."""
    clients = ring_clients(FederatedClient.__name__)
    message = VanillaFederatedMessage(time=0, source_client=ServerMockClient(), global_model=clients["3"].model)

    clients["0"].aggregation([message])

    assert weight(clients["0"]) == 3.0


# --- Engines -----------------------------------------------------------------------

class TestDecentralizedEngines:
    """Engines that run decentralized averaging over a graph."""

    def test_sequential_trains_and_tests_every_client_each_round(self):
        scenario = VanillaDecentralizedSequential(
            **scenario_inputs(ring(4), "DecAvgClient", DEC_AVG), common_init=False, max_communication_rounds=2
        )

        scenario.run()

        for client in scenario.clients.values():
            assert engine_of(client).train_calls == [0, 1]
            assert engine_of(client).test_calls == 2

    def test_one_synch_round_averages_neighbours(self):
        scenario = scenario_with_weights(VanillaDecentralizedSynch(
            **scenario_inputs(ring(4), "DecAvgClient", DEC_AVG), common_init=False, max_communication_rounds=1
        ))

        scenario.run()

        # Each node takes the mean of its two ring neighbours.
        assert [weight(scenario.clients[n]) for n in "0123"] == [2.0, 1.0, 2.0, 1.0]

    def test_parallel_clients_give_the_same_schedule(self):
        scenario = VanillaDecentralizedSynch(
            **scenario_inputs(ring(4), "DecAvgClient", DEC_AVG),
            common_init=False, max_communication_rounds=2, num_parallel_clients=2,
        )

        scenario.run()

        assert all(engine_of(c).train_calls == [0, 1] for c in scenario.clients.values())

    def test_synchronize_sends_a_snapshot_to_neighbours_only(self):
        scenario = VanillaDecentralizedSynch(
            **scenario_inputs(ring(4), "DecAvgClient", DEC_AVG), common_init=False, max_communication_rounds=1
        )
        sender = scenario.clients["0"]

        scenario.synchronize_client(0, sender)
        set_weight(sender, 99.0)  # changes after sending must not leak

        assert [scenario.clients[n].message_buffer.qsize() for n in "0123"] == [0, 1, 0, 1]
        received = scenario.clients["1"].message_buffer.get()
        assert received.source_client is not sender
        assert weight(received.source_client) == 0.0

    def test_shallow_sync_sends_the_client_itself(self):
        scenario = DecentralizedShallowSyncReference(
            **scenario_inputs(ring(4), "DecAvgClient", DEC_AVG), common_init=False, max_communication_rounds=1
        )
        sender = scenario.clients["0"]

        scenario.synchronize_client(0, sender)

        assert scenario.clients["1"].message_buffer.get().source_client is sender


class TestFederatedScenario:
    """``VanillaFederatedScenario``: a server averages all clients after every round."""

    def make(self) -> VanillaFederatedScenario:
        """A federated scenario on 4 clients, weights 0..3."""
        scenario = VanillaFederatedScenario(
            use_weighted_avg=False, **scenario_inputs(ring(4), FederatedClient.__name__),
            common_init=False, max_communication_rounds=1,
        )
        return cast(VanillaFederatedScenario, scenario_with_weights(scenario))

    def test_graph_becomes_a_star_around_the_server(self):
        scenario = self.make()

        nx_graph = cast(Any, scenario.graph).nx_graph
        assert set(nx_graph.neighbors("C")) == {0, 1, 2, 3}
        assert nx_graph.number_of_edges() == 4

    def test_round_end_sends_the_global_average_to_every_client(self):
        scenario = self.make()

        scenario.on_communication_round_end(0)
        for client in scenario.clients.values():
            client.update(1)

        assert all(weight(c) == pytest.approx(1.5) for c in scenario.clients.values())  # mean of 0..3


class TestCentralizedScenario:
    """``CentralizedScenarioEngine``: one client trained once on all the data."""

    def test_trains_and_tests_the_single_client_once(self):
        scenario = CentralizedScenarioEngine(**scenario_inputs(make_graph(graph=nx.empty_graph(1)), "CentralizedClient"))

        scenario.run()

        client = scenario.clients["0"]
        assert engine_of(client).train_calls == [0]
        assert engine_of(client).test_calls == 1

    def test_requires_exactly_one_node(self):
        with pytest.raises(AssertionError):
            CentralizedScenarioEngine(**scenario_inputs(ring(2), "CentralizedClient"))


# --- Utilities -----------------------------------------------------------------

class TestExecutors:
    """``executor_context``: runs client shards serially or on threads."""

    def test_serial_executor_returns_results(self):
        assert SerialExecutor().submit(sum, [1, 2, 3]).result() == 6

    def test_serial_executor_keeps_exceptions_in_the_future(self):
        future = SerialExecutor().submit(int, "not a number")

        with pytest.raises(ValueError):
            future.result()

    def test_one_worker_means_serial(self):
        assert isinstance(executor_context(1), SerialExecutor)
        assert not isinstance(executor_context(2), SerialExecutor)
