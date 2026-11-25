"""Deterministic public event and descriptor behavior."""
import unittest
from datetime import UTC, datetime
from uuid import uuid4

from dengine.scenarios.event_api.events import (
    Event, EventHandlerDescriptor, LocalTraining, LocalTrainingDescriptor,
    ParallelEvent, ParallelEventDescriptor, StartEvent, EndEvent,
    Synchronization, SynchronizationDescriptor,
)


class EventTests(unittest.TestCase):
    def test_explicit_timestamp_ordering(self):
        early = Event(tag='early', timestamp=datetime.fromtimestamp(10, UTC))
        late = Event(tag='late', timestamp=datetime.fromtimestamp(20, UTC))
        tied = Event(tag='tie', timestamp=early.timestamp)
        self.assertLess(early, late)
        self.assertGreater(late, early)
        self.assertFalse(early < tied)
        self.assertFalse(early > tied)
        self.assertEqual([e.tag for e in sorted([late, early])], ['early', 'late'])

    def test_tag_and_uuid_matching(self):
        event = StartEvent()
        other = uuid4()
        for selector, expected in [(event.uuid, True), (other, False),
                                   ([other, event.uuid], True), ((other,), False)]:
            with self.subTest(selector=selector):
                # An explicit UUID selects the exact event independently of its tag.
                descriptor = EventHandlerDescriptor('END', 'start', None, selector)
                self.assertEqual(descriptor.match(event), expected)
        # No tag accepts any event when a UUID is also absent.
        self.assertTrue(EventHandlerDescriptor(None, 'end', None, None).match(event))
        # A supplied tag must match the event.
        self.assertTrue(EventHandlerDescriptor('START', 'start', None, None).match(event))
        self.assertFalse(EventHandlerDescriptor('END', 'start', None, None).match(event))

    def test_timestamp_and_tag_must_both_match(self):
        event = StartEvent(timestamp=datetime.fromtimestamp(10, UTC))
        cases = [('START', 10, True), ('END', 10, False), ('START', 11, False)]
        for tag, timestamp, expected in cases:
            with self.subTest(tag=tag, timestamp=timestamp):
                descriptor = EventHandlerDescriptor(tag, 'start', timestamp, None)
                self.assertEqual(descriptor.match(event), expected)

    def test_local_training_client_filters(self):
        event = LocalTraining(client='alice')
        for client, expected in [(None, True), ('alice', True), ('bob', False),
                                 (['bob', 'alice'], True), (['bob'], False)]:
            with self.subTest(client=client):
                descriptor = LocalTrainingDescriptor(None, 'start', None, None, client)
                self.assertEqual(descriptor.match(event), expected)
        self.assertFalse(LocalTrainingDescriptor('END', 'start', None, None, None).match(event))

    def test_synchronization_filters(self):
        event = Synchronization(src='alice', destinations=['bob', 'carol'])
        for src, dst, expected in [(None, None, True), ('alice', 'bob', True),
                                   (['alice'], ['carol'], True), ('bob', None, False),
                                   (None, ['dan'], False)]:
            with self.subTest(src=src, dst=dst):
                descriptor = SynchronizationDescriptor(None, 'start', None, None, src, dst)
                self.assertEqual(descriptor.match(event), expected)

    def test_parallel_membership(self):
        first, second = StartEvent(), EndEvent()
        event = ParallelEvent(tag='parallel', events=[first, second], num_workers=2)
        other = uuid4()
        for any_ids, all_ids, expected in [
            (first.uuid, None, True), ([other, second.uuid], None, True),
            (other, None, False), (None, [first.uuid, second.uuid], True),
            (None, [first.uuid, other], False), (None, second.uuid, True),
        ]:
            with self.subTest(any_ids=any_ids, all_ids=all_ids):
                descriptor = ParallelEventDescriptor(None, 'start', None, None, any_ids, all_ids)
                self.assertEqual(descriptor.match(event), expected)
        descriptor = ParallelEventDescriptor(None, 'start', None, None, None, None)
        self.assertFalse(descriptor.match(first))

    def test_parallel_validation(self):
        from pydantic import ValidationError
        for events, workers in [([], 1), ([StartEvent()], 0)]:
            with self.subTest(events=events, workers=workers):
                with self.assertRaises(ValidationError):
                    ParallelEvent(tag='parallel', events=events, num_workers=workers)
