from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Optional, List, TypeVar, Generic, Sequence, Literal, Dict, Callable
from queue import PriorityQueue
from dataclasses import dataclass, field
import uuid

import torch
from torch import Tensor
from torch.utils.data import Subset
from torch import nn

from dengine.dataset import SupervisedDataset
from dengine.config import DynamicModuleConfigBase, ClientModuleConfig, ExperimentConfiguration
from dengine.graph import Graph
from dengine.partitioning import TYPE_DATASET_PARTITIONING


GenericClient = TypeVar("GenericClient", bound="ClientInterface")
GenericMessage = TypeVar("GenericMessage", bound="MessageBase")


@dataclass
class MessageBase(Generic[GenericClient]):
    time: float
    source_client: GenericClient
    UUID: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __lt__(self, other: MessageBase):
        return self.time < other.time

    def __eq__(self, other: MessageBase):
        return self.time == other.time


class ClientInterface(ABC, Generic[GenericMessage]):
    @abstractmethod
    def __init__(
        self,
        scenario: ScenarioEngineInterface,
        training_engine: DynamicModuleConfigBase | LocalTrainingEngineInterface,
        local_model: DynamicModuleConfigBase | ModuleBase,
        training_data: Subset,
        validation_data: Subset,
        callback: Optional[ClientCallbackInterface] = None,
        uuid: Optional[str] = None
    ):
        raise NotImplementedError()

    @property
    def UUID(self) -> str:
        raise NotImplementedError()

    @property
    def callback(self) -> ClientCallbackInterface:
        raise NotImplementedError()

    @callback.setter
    def callback(self, value: ClientCallbackInterface):
        raise NotImplementedError()

    @property
    def local_training_engine(self) -> LocalTrainingEngineInterface:
        raise NotImplementedError()

    @property
    def message_buffer(self) -> PriorityQueue[GenericMessage]:
        raise NotImplementedError()

    @property
    def model(self) -> ModuleBase:
        raise NotImplementedError()

    @model.setter
    def model(self, value: ModuleBase):
        raise NotImplementedError()

    @abstractmethod
    def synchronization(self, current_time: int) -> List[GenericMessage]:
        raise NotImplementedError()

    @abstractmethod
    def aggregation(self, messages: List[GenericMessage]):
        raise NotImplementedError()

    @abstractmethod
    def update(self, current_time: float):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def test(self, current_time: float, dataset: SupervisedDataset):
        raise NotImplementedError()


class ScenarioEngineInterface(ABC, Generic[GenericClient]):
    @abstractmethod
    def __init__(
        self,
        graph: Graph,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        test_data: SupervisedDataset,
        client_configuration: ClientModuleConfig,
        callback_factory: Optional[TYPE_CLIENT_CALLBACK_FACTORY] = None
    ):
        raise NotImplementedError()

    @property
    def clients(self) -> Dict[str, GenericClient]:
        raise NotImplementedError()

    @property
    def graph(self) -> Graph:
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()


class LocalTrainingEngineInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        training_data: Subset,
        validation_data: Subset,
        scenario: ScenarioEngineInterface,
        client: ClientInterface,
        *args,
        callback: Optional[ClientCallbackInterface] = None,
        **kwargs
    ):
        raise NotImplementedError()

    @property
    def callback(self) -> ClientCallbackInterface:
        raise NotImplementedError()

    @callback.setter
    def callback(self, value: ClientCallbackInterface):
        raise NotImplementedError()

    @abstractmethod
    def train(self, model: ModuleBase, current_time: float) -> ModuleBase:
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def compute_loss_in_production(
        self,
        model: ModuleBase,
        data: SupervisedDataset
    ) -> Dict[Literal["output", "loss"], Tensor]:
        raise NotImplementedError()


class ModuleBase(nn.Module):
    def __init__(
        self,
        dataset: Subset,
        *args,
        **kwargs
    ):
        super().__init__()


class ClientCallbackInterface(Generic[GenericMessage]):
    @abstractmethod
    def __init__(
        self,
        client: ClientInterface,
        scenario: ScenarioEngineInterface,
        test_data: SupervisedDataset,
        experiment_configuration: ExperimentConfiguration,
    ):
        raise NotImplementedError()

    def on_synchronization_start(self, current_time: float):
        ...

    def on_synchronization_end(self, current_time: float, messages: Sequence[GenericMessage]):
        ...

    def on_aggregation_start(self, current_time: float):
        ...

    def on_aggregation_end(self, current_time: float):
        ...

    def on_local_training_start(self, current_time: float):
        ...

    def on_local_training_end(self, current_time: float, **kwargs):
        ...

    def on_training_batch_start(self, step: int, *args):
        ...

    def on_training_batch_end(self, step: int, *args, **kwargs):
        ...

    def on_training_epoch_start(self, epoch: int):
        ...

    def on_training_epoch_end(self, epoch: int, **kwargs):
        ...

    def on_test_inference_start(self, current_time: float):
        ...

    def on_test_inference_end(self, current_time: float, **kwargs):
        ...


TYPE_CLIENT_CALLBACK_FACTORY = Callable[[ClientInterface, ScenarioEngineInterface], ClientCallbackInterface]
