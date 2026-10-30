from __future__ import annotations

from typing import Sequence, Literal, List
from uuid import UUID
from dataclasses import dataclass

from pydantic import BaseModel, UUID4, Field
from datetime import datetime, UTC
from uuid import uuid4


class Event(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.fromtimestamp(0, UTC))
    uuid: UUID4 = Field(default_factory=uuid4)
    tag: str

    def __lt__(self, e: Event):
        return self.timestamp < e.timestamp

    def __gt__(self, e: Event):
        return self.timestamp > e.timestamp

    def __eq__(self, e: Event):
        return self.uuid == e.uuid

    @classmethod
    def repr(cls) -> str:
        return cls.model_fields["tag"].default  # pylint: disable=unsubscriptable-object


@dataclass
class EventHandlerDescriptor:
    event_tag: str | None
    stage: Literal["start", "end"]
    timestamp: float | None
    uuids: UUID | Sequence[UUID] | None

    def match(self, event: Event) -> bool:
        if self.uuids:
            if (
                isinstance(self.uuids, Sequence)
                and event.uuid in self.uuids
            ):
                return True
            elif (event.uuid == self.uuids):
                return True
            return False

        if self.timestamp:
            return event.timestamp == self.timestamp

        if self.event_tag and self.event_tag != event.tag:
            return False

        return True


class StartEvent(Event):
    tag: Literal["START"] = "START"


class EndEvent(Event):
    tag: Literal["END"] = "END"


class Synchronization(Event):
    src: str
    destinations: Sequence[str] | None
    tag: Literal["synchronization"] = "synchronization"


@dataclass
class SynchronizationDescriptor(EventHandlerDescriptor):
    src: str | List[str] | None
    destinations: str | List[str] | None

    def match(self, event: Synchronization) -> bool:
        if not super().match(event):
            return False

        src_is_matching = False
        if self.src:
            if (
                isinstance(self.src, List)
                and event.src in self.src
            ):
                src_is_matching = True
            elif event.src == self.src:
                src_is_matching = True

        dst_is_matching = False
        if self.destinations and event.destinations:
            if (
                isinstance(self.destinations, List)
                and any([event_dst in self.destinations for event_dst in event.destinations])
            ):
                dst_is_matching = True
            elif (
                isinstance(self.destinations, str)
                and self.destinations in event.destinations
            ):
                dst_is_matching = True

        return (
            (self.src is None or src_is_matching)
            and (self.destinations is None or dst_is_matching)
        )


class LocalTraining(Event):
    client: str
    tag: Literal["local_training"] = "local_training"


@dataclass
class LocalTrainingDescriptor(EventHandlerDescriptor):
    client: str | List[str] | None

    def match(self, event: LocalTraining) -> bool:
        if not super().match(event):
            return False

        if not self.client:
            return True

        if (
            isinstance(self.client, List)
            and event.client in self.client
        ):
            return True
        elif event.client == self.client:
            return True
        return False
