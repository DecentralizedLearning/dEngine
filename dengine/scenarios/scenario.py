from __future__ import annotations

from abc import abstractmethod
from typing import Optional, List, Dict, Generic, cast
from queue import PriorityQueue
import logging
from copy import deepcopy
from uuid import uuid4
from contextlib import contextmanager

import torch
from torch.utils.data import Subset

from dengine.callbacks import DummyCallback
from dengine.config import (
    ClientModuleConfig,
    DynamicModuleConfigBase,
    instantiate_configuration_module
)
from dengine.dataset import SupervisedDataset
from dengine.dataset.utils import get_targets
from dengine.graph import Graph
from dengine.interfaces import (
    ClientCallbackInterface,
    ClientInterface,
    GenericClient,
    GenericMessage,
    LocalTrainingEngineInterface,
    ModuleBase,
    ScenarioEngineInterface,
    TYPE_CLIENT_CALLBACK_FACTORY
)
from dengine.models.decorators import BUILTIN_MODELS
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.training_strategies.decorators import BUILTIN_TRAINING_ENGINES
from dengine.utils.utils import tensor_report

from .decorators import BUILTIN_CLIENTS


class AbstractClient(ClientInterface[GenericMessage]):
    def __init__(
        self,
        scenario: AbstractScenarioEngine,
        training_engine: DynamicModuleConfigBase | LocalTrainingEngineInterface,
        local_model: DynamicModuleConfigBase | ModuleBase,
        training_data: Subset,
        validation_data: Subset,
        callback: Optional[ClientCallbackInterface] = None,
        # Additional kwargs
        uuid: Optional[str] = None,
        verbose: bool = False,
        force_time_synchronization: bool = True,
        debug_skip_training: bool = False
    ):
        self._scenario = scenario
        self._callback = callback or DummyCallback()
        if isinstance(local_model, DynamicModuleConfigBase):
            self._model: ModuleBase = instantiate_configuration_module(
                config=local_model,
                allowed_cls=BUILTIN_MODELS,
                superclass=ModuleBase,
                callback=self._callback,
                dataset=training_data,
            )
        else:
            self._model = local_model
        if isinstance(training_engine, DynamicModuleConfigBase):
            self._local_strategy: LocalTrainingEngineInterface = instantiate_configuration_module(
                config=training_engine,
                allowed_cls=BUILTIN_TRAINING_ENGINES,
                superclass=LocalTrainingEngineInterface,
                training_data=training_data,
                validation_data=validation_data,
                scenario=scenario,
                client=self,
                callback=self._callback,
            )
        else:
            self._local_strategy = training_engine

        self._uuid = uuid4() if uuid is None else uuid
        self.verbose = verbose
        self._force_time_synchronization = force_time_synchronization
        self._debug_skip_training = debug_skip_training

        self._msg_buffer: PriorityQueue[GenericMessage] = PriorityQueue()

        dataset_train_report = tensor_report(get_targets(training_data), "**Train**")
        dataset_valid_report = tensor_report(get_targets(validation_data), "**Validation**")
        self._logging(
            f"### {self.__class__.__name__}-{self.UUID} Initialized\n"
            f"{dataset_train_report} \n"
            f"{dataset_valid_report} \n"
            f"---"
        )

    @property
    def model(self) -> ModuleBase:
        return self._model

    @model.setter
    def model(self, value: ModuleBase):
        self._model = value

    def load_state_dict(self, model_update):
        self.model.load_state_dict(model_update)

    @property
    def _logging(self):
        def mock_call(s: str):
            pass

        if not self.verbose:
            return mock_call
        return logging.info

    @property
    def UUID(self) -> str:
        return str(self._uuid)

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value
        self._local_strategy.callback = value

    @property
    def message_buffer(self) -> PriorityQueue[GenericMessage]:
        return self._msg_buffer

    @property
    def local_training_engine(self) -> LocalTrainingEngineInterface:
        return self._local_strategy

    @contextmanager
    def on_device(self, device: torch.device):
        self.model = self.model.to(device)
        yield
        self.model = self.model.cpu()

    def synchronization(self, current_time: float) -> List[GenericMessage]:
        if self.message_buffer.empty():
            return []

        messages: List[GenericMessage] = []
        items_to_reinsert: List[GenericMessage] = []
        while not self.message_buffer.empty():
            msg = self.message_buffer.get()
            if self._force_time_synchronization and (msg.time > current_time):
                items_to_reinsert.append(msg)
                continue
            messages.append(msg)

        for msg in items_to_reinsert:
            self.message_buffer.put(msg)

        filtered_messages: Dict[str, GenericMessage] = {}
        for m in messages:
            if m.source_client.UUID not in filtered_messages:
                filtered_messages[m.source_client.UUID] = m
                continue
            prev_msg = filtered_messages[m.source_client.UUID]
            if prev_msg.time < m.time:
                filtered_messages[m.source_client.UUID] = m

        return list(filtered_messages.values())

    def update(self, current_time: float):
        # 1. Perform synchronization
        self._callback.on_synchronization_start(current_time)
        messages = self.synchronization(current_time)
        self._callback.on_synchronization_end(current_time, messages)

        # 2. Aggregation
        self._callback.on_aggregation_start(current_time)
        with torch.no_grad():
            self.aggregation(messages)
        self._callback.on_aggregation_end(current_time)

        # 3. Train local model
        if self._debug_skip_training:
            return
        self.model = self.execute_local_train_strategy(current_time)

    def execute_local_train_strategy(self, current_time: float):
        """
        Separated for inheritance purposes.

        Decouples local strategy execution from decentralized training, enabling subclasses
        to customize how the local strategy is invoked.

        In the decentralized regime, the actual training involves synchronization and aggregation
        across peers.
        """
        return self._local_strategy.train(self.model, current_time)

    @torch.no_grad()
    def test(self, current_time: float, dataset: SupervisedDataset):
        self._callback.on_test_inference_start(current_time)
        output = self.local_training_engine.compute_loss_in_production(self.model, dataset)
        self._callback.on_test_inference_end(
            current_time=current_time,
            output=output
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ("_scenario", "_callback", "_local_strategy", "_msg_buffer"):
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def aggregation(self, messages: List[GenericMessage]):
        raise NotImplementedError()


class AbstractScenarioEngine(ScenarioEngineInterface, Generic[GenericClient]):
    def __init__(
        self,
        graph: Graph,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        test_data: SupervisedDataset,
        client_configuration: ClientModuleConfig,
        # Additional args
        common_init: bool,
        callback_factory: Optional[TYPE_CLIENT_CALLBACK_FACTORY] = None,
    ):
        self._graph = graph
        self.training_data = training_data
        self.test_data = test_data
        self.partitioning = data_partitions
        self._clients = self.init_clients(
            client_configuration,
            training_data,
            data_partitions,
            common_init,
            callback_factory
        )

    @property
    def graph(self) -> Graph:
        return self._graph

    def init_clients(
        self,
        client_configuration: ClientModuleConfig,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        common_init: bool,
        callback_factory: Optional[TYPE_CLIENT_CALLBACK_FACTORY] = None,
    ) -> Dict[str, GenericClient]:
        local_model = None
        clients: Dict[str, GenericClient] = {}
        for pid in list(self.graph.nodes):
            pid = str(pid)

            if local_model and common_init:
                local_model = deepcopy(local_model)
            else:
                local_model = client_configuration.local_model

            _client_partition = data_partitions[pid]
            _client = instantiate_configuration_module(
                config=client_configuration,
                superclass=ClientInterface,
                allowed_cls=BUILTIN_CLIENTS,
                scenario=self,
                training_engine=client_configuration.training_engine,
                local_model=local_model,
                training_data=Subset(training_data, _client_partition["train"].tolist()),
                validation_data=Subset(training_data, _client_partition["validation"].tolist()),
                uuid=pid,
            )
            local_model = _client.model

            if callback_factory:
                _client.callback = callback_factory(_client, self)
            clients[_client.UUID] = cast(GenericClient, _client)
        return clients

    @property
    def clients(self) -> Dict[str, GenericClient]:
        return self._clients

    @abstractmethod
    def run(self):
        raise NotImplementedError()
