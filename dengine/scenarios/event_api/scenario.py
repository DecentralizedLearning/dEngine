from __future__ import annotations

"""
All inspiration for the following pattern was given by FastAPI, Pydantic and wouldn't
have been possible without https://ponderinglion.dev/posts/demystifying-fastapis-dependency-injection/
"""

import logging
import sys
from uuid import UUID
from abc import abstractmethod
from datetime import datetime, UTC
from collections import defaultdict
from queue import Empty
import inspect
from inspect import Signature
from typing import Callable, Tuple, List, Any, Literal, Sequence, Optional, Dict
import functools
from dataclasses import dataclass

from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.config import ClientModuleConfig
from dengine.graph import Graph
from dengine.dataset import SupervisedDataset
from dengine.scenarios.scenario import GenericClient, TYPE_CLIENT_CALLBACK_FACTORY, AbstractScenarioEngine
from dengine.scenarios.decorators import register_scenario
import dengine.scenarios.event_api.events as e
from dengine.scenarios.event_api.events import (
    Event,
    EventHandlerDescriptor,
    LocalTrainingDescriptor,
    SynchronizationDescriptor
)


@dataclass
class Handler:
    descriptor: EventHandlerDescriptor
    call: Callable[[Event], Sequence[Event] | None]


class Dependency:
    def __init__(
        self,
        dependency: Callable[..., Any],
        *args, **kwargs
    ):
        self._dependency = dependency
        args = args
        kwargs = kwargs

    def resolve(self, event: Event, engine: ScenarioEventEngine):
        signature = inspect.signature(self._dependency)
        resolved_kwargs = {
            param_name: resolve_if_dependency(param, event, engine)
            for param_name, param in signature.parameters.items()
        }
        resolved_kwargs.update(
            resolve_events_and_engine(signature, event=event, engine=engine)
        )
        return self._dependency(**resolved_kwargs)


def Depends(
    dependency: Callable[..., Any],
    *args, **kwargs
) -> Any:
    return Dependency(dependency=dependency, *args, **kwargs)


def resolve_events_and_engine(
    signature: Signature,
    event: Event,
    engine: ScenarioEventEngine
) -> Dict:
    events = {}
    for k, p in signature.parameters.items():
        try:
            if (
                (isinstance(p.annotation, str) and p.annotation == "Event") or
                issubclass(p.annotation, Event)
            ):
                events[k] = event
            if issubclass(p.annotation, ScenarioEventEngine):
                events[k] = engine
        except TypeError:
            pass
    return events


def resolve_if_dependency(param: inspect.Parameter, event: Event, engine: ScenarioEventEngine):
    """Extracts the dependency object from the parameter annotation"""
    if param.default and isinstance(param.default, Dependency):
        return param.default.resolve(event, engine)
    return param


def format_lap(lap: datetime):
    total_seconds = (datetime.now(tz=UTC) - lap).total_seconds()
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {seconds:.2f}s"


@register_scenario()
class ScenarioEventEngine(AbstractScenarioEngine[GenericClient]):
    def __init__(
        self,
        max_communication_rounds: int = sys.maxsize,
        verbose: bool = True,
    ):
        self.timestamp = datetime.fromtimestamp(0, tz=UTC)
        self._max_communication_rounds = max_communication_rounds
        self._verbose = verbose
        self._event_handlers: defaultdict[
            Tuple[str | None, str],  # [tag, phase]
            List[Handler]
        ] = defaultdict(list)

    @abstractmethod
    def load(
        self,
        graph: Graph,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        test_data: SupervisedDataset,
        client_configuration: ClientModuleConfig,
        callback_factory: Optional[TYPE_CLIENT_CALLBACK_FACTORY] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def pop_next_event(self) -> e.Event:
        raise NotImplementedError()

    @abstractmethod
    def add_events(self, events: e.Event | Sequence[e.Event]):
        raise NotImplementedError()

    @property
    def _logging(self):
        def mock_call(s: str):
            pass

        if not self._verbose:
            return mock_call
        return logging.info

    def _format_event_note(self, s: str) -> str:
        return f"• {s}"

    def _format_event_str(self, event: Event) -> str:
        cls_name = event.__class__.__name__
        fields = event.model_dump()
        fields['timestamp'] = event.timestamp.timestamp()
        formatted_fields = "\n".join(
            self._format_event_note(f"{k}: {v}")
            for k, v in fields.items()
        )
        return f"{cls_name}\n{formatted_fields}"

    def logging_note(self, s: str):
        self._logging(self._format_event_note(s))

    def run(self):
        self.add_events(e.StartEvent())
        try:
            while True:
                next_event = self.pop_next_event()

                self._logging(self._format_event_str(next_event))
                lap = datetime.now(tz=UTC)

                assert next_event.timestamp >= self.timestamp
                self.timestamp = next_event.timestamp
                if next_event.timestamp.timestamp() >= self._max_communication_rounds:
                    break
                self._consume_event_with_handlers(next_event)

                self.logging_note(f"Total time: {format_lap(lap)}")
                self._logging("")
        except Empty:
            logging.info("No more events to consume")

    def _consume_event_with_handlers(self, event: Event):
        start_event_handlers = filter(
            lambda h: h.descriptor.match(event),
            self._event_handlers[(event.tag, "start")]
        )
        start_event_handlers = list(start_event_handlers) + self._event_handlers[(None, "start")]
        for h in start_event_handlers:
            self.logging_note(f"{h.call.__name__}()")
            h.call(event)

        self.consume_event(event)

        end_event_handlers = filter(
            lambda h: h.descriptor.match(event),
            self._event_handlers[(event.tag, "end")]
        )
        end_event_handlers = list(end_event_handlers) + self._event_handlers[(None, "end")]
        for h in end_event_handlers:
            self.logging_note(f"{h.call.__name__}()")
            h.call(event)

    @abstractmethod
    def consume_event(self, event: Event):
        raise NotImplementedError

    def register_event_handler(
        self,
        descriptor: EventHandlerDescriptor,
    ):
        def wrapper(func: Callable[..., Sequence[Event] | None]) -> Callable:
            @functools.wraps(func)
            def call_with_dependencies(event: Event, *args, **kwargs) -> Any:
                signature = inspect.signature(func)
                resolved_kwargs = {
                    param_name: resolve_if_dependency(param, event, self)
                    for param_name, param in signature.parameters.items()
                }
                resolved_kwargs.update(
                    resolve_events_and_engine(signature, event=event, engine=self)
                )
                new_events = func(*args, **resolved_kwargs)
                if new_events:
                    self.add_events(new_events)
                return new_events

            handler = Handler(descriptor, call_with_dependencies)
            self._event_handlers[descriptor.event_tag, descriptor.stage].append(handler)

            return call_with_dependencies
        return wrapper

    def start(self):
        descriptor = EventHandlerDescriptor(
            event_tag=e.StartEvent.repr(),
            stage='start',
            timestamp=None,
            uuids=None,
        )
        return self.register_event_handler(descriptor)

    def synchronization(
        self,
        stage: Literal["start", "end"],
        timestamp: float | None = None,
        uuids: UUID | Sequence[UUID] | None = None,
        src: UUID | Sequence[UUID] | None = None,
        destination: UUID | Sequence[UUID] | None = None,
    ):
        descriptor = SynchronizationDescriptor(
            event_tag=e.Synchronization.repr(),
            stage=stage,
            timestamp=timestamp,
            uuids=uuids,
            src=src,
            destinations=destination
        )
        return self.register_event_handler(descriptor)

    def local_training(
        self,
        stage: Literal["start", "end"],
        timestamp: float | None = None,
        uuids: UUID | Sequence[UUID] | None = None,
        client: UUID | Sequence[UUID] | None = None,
    ):
        descriptor = LocalTrainingDescriptor(
            event_tag=e.LocalTraining.repr(),
            stage=stage,
            timestamp=timestamp,
            uuids=uuids,
            client=client
        )
        return self.register_event_handler(descriptor)

    def user_event(
        self,
        tag: str | None = None,
        stage: Literal["start", "end"] = "start",
        timestamp: float | None = None,
        uuids: UUID | Sequence[UUID] | None = None,
        client: UUID | Sequence[UUID] | None = None,
    ):
        descriptor = LocalTrainingDescriptor(
            event_tag=tag,
            stage=stage,
            timestamp=timestamp,
            uuids=uuids,
            client=client
        )
        return self.register_event_handler(descriptor)

    # Main interface
    # ..... ..... ..... ..... #
    @abstractmethod
    def get_client(
        self,
        uuids: str
    ) -> GenericClient:
        raise NotImplementedError()

    @abstractmethod
    def get_many_clients(
        self,
        uuids: str | Sequence[str] | None
    ) -> Sequence[GenericClient]:
        raise NotImplementedError()
