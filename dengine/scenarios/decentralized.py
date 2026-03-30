from __future__ import annotations

from typing import List, Sequence, TypeVar
from abc import abstractmethod
from warnings import deprecated

from copy import deepcopy
import logging
from dataclasses import dataclass

import torch
from torch.utils.data import Subset

from dengine.utils.utils import model_on_device_context
from dengine.models.utils import model_wise_weighted_average
from dengine.interfaces import MessageBase

from .scenario import AbstractClient, AbstractScenarioEngine
from .decorators import register_scenario, register_client


@dataclass
class VanillaDecentralizedMessage(MessageBase):
    source_client: DecAvgClient


TGenericVanillaDecentralizedMessage = TypeVar("TGenericVanillaDecentralizedMessage", bound=VanillaDecentralizedMessage)


def _client_wise_alpha_weighted_avg(
    messages: Sequence[MessageBase[DecAvgClient]],
    alpha_list: Sequence[float]
):
    '''
    Computes the weighted federated averaged of the models
    received from the paiv's social graph, weighted by their social trust
    '''
    return model_wise_weighted_average(
        [m.source_client.model.state_dict() for m in messages],
        alpha_list
    )


@register_client()
class DecAvgClient(AbstractClient[TGenericVanillaDecentralizedMessage]):
    def __init__(
        self,
        training_data: Subset,
        *args,
        use_weighted_avg: bool,
        selfconfidence: float = 1,
        include_myself: bool = False,
        **kwargs
    ):
        super().__init__(*args, training_data=training_data, **kwargs)
        self._include_myself = include_myself
        self._selfconfidence = selfconfidence
        self._use_weighted_avg = use_weighted_avg
        self._graph = self._scenario.graph
        self.train_size = len(training_data)

    def _get_alphas(
        self,
        messages: Sequence[TGenericVanillaDecentralizedMessage | VanillaDecentralizedMessage],
    ) -> Sequence[float]:
        tot_models = len(messages)
        tot_data_neighborhood = sum([_msg.source_client.train_size for _msg in messages])

        messages_alpha = []
        for msg in messages:
            if self._use_weighted_avg:
                alpha = msg.source_client.train_size / tot_data_neighborhood
            else:
                alpha = 1. / tot_models
            messages_alpha.append(alpha)
        return messages_alpha

    def aggregation(self, messages: List[TGenericVanillaDecentralizedMessage]):
        messages = self.filter_message_from_neighborhood(messages)
        if len(messages) == 0:
            return

        message_origins = [_p.source_client.UUID for _p in messages]
        message_times = [_p.time for _p in messages]
        origins_with_times = [f"{src} at {t}" for src, t in zip(message_origins, message_times)]
        logging.info(
            f"PAIV-{self.UUID}: Aggregating {len(messages)} models. "
            f"Sources/Timestamps: [{', '.join(origins_with_times)}]"
        )

        if self._include_myself:
            messages.append(
                VanillaDecentralizedMessage(time=messages[0].time, source_client=self)  # type: ignore
            )

        model_avg = _client_wise_alpha_weighted_avg(
            messages,
            self._get_alphas(messages)
        )
        self.model.load_state_dict(model_avg)

    def filter_message_from_neighborhood(
        self,
        messages: Sequence[TGenericVanillaDecentralizedMessage]
    ) -> List[TGenericVanillaDecentralizedMessage]:
        neighbors = list(self._graph.neighbors(self.UUID))
        return [m for m in messages if m.source_client.UUID in neighbors]

    def get_trust_in_neighs(self):
        neighbors = list(self._graph.neighbors(self.UUID))
        trust_dict = {
            n: self._graph.get_weight(self.UUID, n)
            for n in neighbors
        }
        return trust_dict

    def get_dst_list(self):
        return self._graph.neighbors(self.UUID)


@deprecated("This class has been refactored to DecAvgClient, please consider refactoring.")
@register_client()
class FedAvgPaiv(DecAvgClient):
    def __init__(
        self,
        *args,
        test_batch_size,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


@register_scenario()
class DecentralizedScenarioEngineBase(AbstractScenarioEngine[DecAvgClient]):
    def __init__(
        self,
        *args,
        max_communication_rounds: int,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._max_communication_rounds = max_communication_rounds
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

    def get_client_by_id(self, id) -> DecAvgClient:
        return self.clients[id]

    def get_active_clients(self) -> Sequence[DecAvgClient]:
        return list(self.clients.values())

    def run(self):
        for round in range(self._max_communication_rounds):
            logging.info(f"# Starting communication round **{round}**")
            for p in self.get_active_clients():
                self._client_step(round, p)
            self.on_communication_round_end(round)

    def on_communication_round_end(self, round_idx: int):
        pass

    def _client_step(self, communication_round: int, client: DecAvgClient):
        logging.info(
            f"**[Client-{client.UUID}][CommunicationRound-{communication_round}]** \n"
            "```"
        )
        with model_on_device_context(client.model, self._device):
            client.update(communication_round)
            logging.info("```\n```")
            client.test(communication_round, self.test_data)
            logging.info("```")
        logging.info("```")
        self.synchronize_client(communication_round, client)
        logging.info("```")

    @abstractmethod
    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        raise NotImplementedError()


@register_scenario()
class VanillaDecentralizedSequential(DecentralizedScenarioEngineBase):
    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = list(client.get_dst_list())
        logging.info(f"Sending the local model to: {destinations_idxs}")

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client_checkpoint)
            )


@register_scenario()
class DecentralizedShallowSyncReference(DecentralizedScenarioEngineBase):
    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = list(client.get_dst_list())
        logging.info(f"Sending the local model to: {destinations_idxs}")

        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client)
            )


@register_scenario()
class VanillaDecentralizedSynch(DecentralizedScenarioEngineBase):
    def run(self):
        for _client in self.clients.values():
            _client.model.cpu()
            self.synchronize_client(-1, _client)

        for _round in range(self._max_communication_rounds):
            for _client in self.get_active_clients():
                with model_on_device_context(_client.model, self._device):
                    _client.update(_round)
                self.synchronize_client(_round, _client)

            for _client in self.get_active_clients():
                _client.test(_round, self.test_data)

            self.on_communication_round_end(_round)

    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = list(client.get_dst_list())

        title = f"[Paiv-{client.UUID}][CommunicationRound {communication_round}]"
        logging.info(f"{title} Sending the local model to: {destinations_idxs}")

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client_checkpoint)
            )
