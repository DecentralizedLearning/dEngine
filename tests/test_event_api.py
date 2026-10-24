"""Tests for the event-driven scenario API: events, handler matching and ``SyncEngine``."""
from datetime import UTC, datetime
from queue import PriorityQueue
from typing import Any, List

import pytest

from dengine.interfaces import ClientInterface
from dengine.scenarios import Depends
from dengine.scenarios.event_api.events import (
    EndEvent,
    Event,
    EventHandlerDescriptor,
    LocalTraining,
    Synchronization,
)
from dengine.scenarios.event_api.sync_engine import SyncEngine, get_training_client

from tests.helpers import engine_of, ring, scenario_inputs


def at(seconds: float) -> datetime:
    """A UTC timestamp ``seconds`` after the epoch."""
    return datetime.fromtimestamp(seconds, UTC)


def make_engine(**options: Any) -> SyncEngine:
    """A ``SyncEngine`` loaded with 4 DecAvg clients on a ring."""
    engine = SyncEngine(**options)
    engine.load(**scenario_inputs(ring(4), "DecAvgClient", {"use_weighted_avg": False}))
    return engine


def buffered(engine: SyncEngine) -> List[int]:
    """How many messages each client ("0".."3") is holding."""
    return [engine.get_client(n).message_buffer.qsize() for n in "0123"]


# --- Events and descriptors ---------------------------------------------------

def test_events_are_ordered_by_timestamp():
    """The engine's priority queue pops the earliest event first."""
    queue: PriorityQueue[Event] = PriorityQueue()
    late, early = EndEvent(timestamp=at(2)), LocalTraining(client="0", timestamp=at(1))
    queue.put(late)
    queue.put(early)

    assert queue.get() is early


class TestEventHandlerDescriptor:
    """``EventHandlerDescriptor.match``: decides which handlers see an event."""

    def test_matches_by_tag(self):
        descriptor = EventHandlerDescriptor(event_tag="local_training", stage="start", timestamp=None, uuids=None)

        assert descriptor.match(LocalTraining(client="0"))
        assert not descriptor.match(EndEvent())

    def test_matches_by_event_uuid(self):
        wanted, other = LocalTraining(client="0"), LocalTraining(client="0")
        descriptor = EventHandlerDescriptor(event_tag=None, stage="start", timestamp=None, uuids=wanted.uuid)

        assert descriptor.match(wanted)
        assert not descriptor.match(other)


# --- SyncEngine -----------------------------------------------------------------

class TestSyncEngine:
    """``SyncEngine``: consumes events in time order until an ``EndEvent``."""

    def test_local_training_trains_and_tests_only_that_client(self):
        engine = make_engine()
        engine.add_events([LocalTraining(client="2", timestamp=at(1)), EndEvent(timestamp=at(2))])

        engine.run()

        assert [engine_of(engine.get_client(n)).train_calls for n in "0123"] == [[], [], [1.0], []]
        assert engine_of(engine.get_client("2")).test_calls == 1

    def test_manual_testing_mode_skips_tests(self):
        engine = make_engine(testing_mode="manual")
        engine.add_events([LocalTraining(client="2", timestamp=at(1)), EndEvent(timestamp=at(2))])

        engine.run()

        assert engine_of(engine.get_client("2")).test_calls == 0

    def test_synchronization_delivers_to_given_destinations(self):
        engine = make_engine()
        engine.add_events([
            Synchronization(src="0", destinations=["1", "3"], timestamp=at(1)),
            EndEvent(timestamp=at(2)),
        ])

        engine.run()

        assert buffered(engine) == [0, 1, 0, 1]

    def test_manual_synchronization_mode_sends_nothing(self):
        engine = make_engine(synchronization_mode="manual")
        engine.add_events([
            Synchronization(src="0", destinations=["1"], timestamp=at(1)),
            EndEvent(timestamp=at(2)),
        ])

        engine.run()

        assert buffered(engine) == [0, 0, 0, 0]

    @pytest.mark.xfail(strict=True, reason="Bug: get_many_clients(None) returns [], so 'all neighbours' becomes 'nobody'")
    def test_synchronization_without_destinations_reaches_neighbours(self):
        engine = make_engine()
        engine.add_events([Synchronization(src="0", destinations=None, timestamp=at(1)), EndEvent(timestamp=at(2))])

        engine.run()

        assert buffered(engine) == [0, 1, 0, 1]

    def test_events_after_the_end_are_ignored(self):
        engine = make_engine()
        engine.add_events([EndEvent(timestamp=at(1)), LocalTraining(client="0", timestamp=at(2))])

        engine.run()

        assert engine_of(engine.get_client("0")).train_calls == []


def test_handlers_receive_dependencies():
    """A handler can ask for the trained client with ``Depends(get_training_client)``."""
    engine = make_engine()
    seen: List[str] = []

    @engine.local_training("end")
    def remember(event: LocalTraining, client: ClientInterface = Depends(get_training_client)) -> None:
        seen.append(client.UUID)

    engine.add_events([
        LocalTraining(client="1", timestamp=at(1)),
        LocalTraining(client="3", timestamp=at(2)),
        EndEvent(timestamp=at(3)),
    ])
    engine.run()

    assert seen == ["1", "3"]
