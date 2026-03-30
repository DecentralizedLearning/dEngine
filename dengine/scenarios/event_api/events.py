from __future__ import annotations

from typing import Sequence, Literal
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


class Synchronization(Event):
    src: str
    destinations: Sequence[str] | None
    tag: Literal["synchronization"] = "synchronization"


@dataclass
class SynchronizationDescriptor(EventHandlerDescriptor):
    src: UUID | Sequence[UUID] | None
    destinations: UUID | Sequence[UUID] | None

    def match(self, event: Synchronization) -> bool:
        if not super().match(event):
            return False

        if self.src:
            if (
                isinstance(self.src, Sequence)
                and event.src in self.src
            ):
                return True
            elif event.src == self.src:
                return True
            return False

        if self.destinations:
            if (
                isinstance(self.destinations, Sequence)
                and event.destinations in self.destinations
            ):
                return True
            elif event.destinations == self.destinations:
                return True
            return False
        return True


class LocalTraining(Event):
    client: str
    tag: Literal["local_training"] = "local_training"


@dataclass
class LocalTrainingDescriptor(EventHandlerDescriptor):
    client: UUID | Sequence[UUID] | None

    def match(self, event: LocalTraining) -> bool:
        if not super().match(event):
            return False

        if not self.client:
            return True

        if (
            isinstance(self.client, Sequence)
            and event.client in self.client
        ):
            return True
        elif event.client == self.client:
            return True
        return False
