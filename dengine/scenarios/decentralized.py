from __future__ import annotations

from typing import List, Sequence, TypeVar
from tqdm import tqdm
import contextlib
from abc import abstractmethod
from warnings import deprecated

from copy import deepcopy
import logging
from dataclasses import dataclass

import torch
from torch.utils.data import Subset

from dengine.models.utils import model_wise_weighted_average
from dengine.interfaces import MessageBase

from .utils import client_on_device_context, executor_context
from .scenario import AbstractClient, AbstractScenarioEngine
from .decorators import register_scenario, register_client


@dataclass
class VanillaDecentralizedMessage(MessageBase):
    source_client: DecAvgClient


TGenericVanillaDecentralizedMessage = TypeVar("TGenericVanillaDecentralizedMessage", bound=VanillaDecentralizedMessage)


@register_client()
class DecAvgClient(AbstractClient[TGenericVanillaDecentralizedMessage]):
    def __init__(
        self,
        training_data: Subset,
        *args,
        use_weighted_avg: bool,
        include_myself: bool = False,
        **kwargs
    ):
        super().__init__(*args, training_data=training_data, **kwargs)
        self._include_myself = include_myself
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
        self.raise_on_non_neighbor_messages(messages)
        if len(messages) == 0:
            return

        if self._include_myself:
            messages.append(
                VanillaDecentralizedMessage(time=messages[0].time, source_client=self)  # type: ignore
            )

        model_avg = model_wise_weighted_average(
            [m.source_client.model.state_dict() for m in messages],
            self._get_alphas(messages)
        )
        self.model.load_state_dict(model_avg)

    def raise_on_non_neighbor_messages(
        self,
        messages: Sequence[TGenericVanillaDecentralizedMessage]
    ):
        neighbors = list(self._graph.neighbors(self.UUID))
        if any(m.source_client.UUID not in neighbors for m in messages):
            raise ValueError("Received one or more messages from a non-neighboring client.")


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

    def get_neighbors(self, client: AbstractClient):
        return list(self._graph.neighbors(client.UUID))

    def on_communication_round_end(self, round_idx: int):
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        raise NotImplementedError()


def log_client_synchronization(
    client: DecAvgClient,
    communication_round: float,
    destinations: List
):
    title = f"[{client.UUID}][CommunicationRound {communication_round}]"
    logging.info(f"{title} Sending the local model to: {destinations}")


@register_scenario()
class VanillaDecentralizedSequential(DecentralizedScenarioEngineBase):
    def run(self):
        clients = list(self.clients.values())
        for round in range(self._max_communication_rounds):
            for client in clients:
                with client_on_device_context(client, self._device):
                    client.update(round)
                    client.test(round, self.test_data)
                self.synchronize_client(round, client)
            self.on_communication_round_end(round)

    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = self.get_neighbors(client)
        log_client_synchronization(client, communication_round, destinations_idxs)

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client_checkpoint)
            )


@deprecated("This class has been refactored to VanillaDecentralizedSequential, please consider refactoring.")
@register_scenario()
class VanillaDecentralized(VanillaDecentralizedSequential):
    pass


@register_scenario()
class VanillaDecentralizedSynch(DecentralizedScenarioEngineBase):
    def __init__(
        self,
        *args,
        num_parallel_clients: int = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._num_parallel_clients = max(num_parallel_clients, 1)

    def run(self):
        clients = list(self.clients.values())

        # for _client in self.clients.values():
        #     _client.model.cpu()
        #     self.synchronize_client(-1, _client)

        pbar = tqdm(range(self._max_communication_rounds), desc="Communication round")

        with executor_context(self._num_parallel_clients) as pool:
            for _round in pbar:
                shards = [
                    clients[i::self._num_parallel_clients]
                    for i in range(self._num_parallel_clients)
                ]
                futures = [
                    pool.submit(self._clients_shard_update_and_test, _round, shard)
                    for shard in shards if shard
                ]
                [f.result() for f in futures]

                futures = [
                    pool.submit(self._clients_shard_synchronize, _round, shard)
                    for shard in shards if shard
                ]
                [f.result() for f in futures]

                self.on_communication_round_end(_round)

    def _clients_shard_update_and_test(self, _round: int, clients: List[DecAvgClient]):
        device = torch.device(self._device)

        if device.type == 'cuda':
            stream = torch.cuda.Stream(device=device)
            stream_context = torch.cuda.stream(stream)
        else:
            stream_context = contextlib.nullcontext()

        with stream_context:
            for _client in clients:
                with client_on_device_context(_client, device):
                    _client.update(_round)
                    _client.test(_round, self.test_data)

    def _clients_shard_synchronize(self, _round: int, clients: List[DecAvgClient]):
        for _client in clients:
            self.synchronize_client(_round, _client)

    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = self.get_neighbors(client)
        log_client_synchronization(client, communication_round, destinations_idxs)

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client_checkpoint)
            )


@register_scenario()
class DecentralizedShallowSyncReference(VanillaDecentralizedSynch):
    def synchronize_client(self, communication_round: int, client: DecAvgClient):
        destinations_idxs = self.get_neighbors(client)
        log_client_synchronization(client, communication_round, destinations_idxs)

        for dst_id in destinations_idxs:
            dst = self.get_client_by_id(dst_id)
            dst.message_buffer.put(
                VanillaDecentralizedMessage(time=communication_round, source_client=client)
            )


@register_client()
class GossipClient(AbstractClient[VanillaDecentralizedMessage]):
    r"""Implementation of the more classic Gossip learning algorithm for distributed averaging usually found in literature (e.g. [1])

    The following implementation is analogous to [2] and simply consist in rescaling the clients' weights accordingly as shown below.
    The original Exact Gossip formulation was taken from [3].

    \begin{align*}
    w_i^{(t)} \leftarrow & w_i^{(t-1)} + \gamma \sum_{j \neq i} p_{ij} \Delta_{ij}^{(t-1)} \\
                         & = w_i^{(t-1)} + \gamma \sum_{j \neq i} p_{ij} \left( w_j^{(t-1)} - w_i^{(t-1)} \right) \\
                         & = - \gamma p_{ii} w_i^{(t-1)} + \gamma \sum_{j \neq i} \left( p_{ij} w_j^{(t-1)} - p_{ij} w_i^{(t-1)} \right) \\
                         & = \gamma \left[ \left( \sum_{j \neq i} p_{ij} w_j^{(t-1)} \right) - \left( \sum_{j \neq i} p_{ij} w_i^{(t-1)} \right) - p_{ii} w_i^{(t-1)} \right] \\
                         & = \gamma \left[ \left( \sum_{j \neq i} p_{ij} w_j^{(t-1)} \right) - w_i^{(t-1)} \sum_j p_{ij} \right] \\
                         & = \gamma \sum_j \hat{p}_{ij} w_j^{(t-1)}
    \end{align*}

    \begin{equation*}
    p_{ii} = - \frac{1}{\gamma} \quad \hat{p}_{ij} =
    \begin{cases}
    p_{ij} & \text{if } i \neq j \\
    \frac{1}{\gamma} - \sum_{i \neq j} p_{ij}
    \end{cases}
    \end{equation*}

    References:
        [1] Xiao, Lin, and Stephen Boyd. "Fast linear iterations for distributed averaging." Systems & Control Letters 53.1 (2004): 65-78.
        [2] Sun, Tao, Dongsheng Li, and Bao Wang. "Decentralized federated averaging." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.4 (2022): 4289-4301.
        [3] Koloskova, Anastasia, Sebastian Stich, and Martin Jaggi. "Decentralized stochastic optimization and gossip algorithms with compressed communication." International conference on machine learning. PMLR, 2019.
    """
    def __init__(
        self,
        *args,
        gamma: float = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._gamma = gamma

    @torch.no_grad()
    def aggregation(self, messages: List[VanillaDecentralizedMessage]):
        if len(messages) == 0:
            return

        for layer_key, layer_W in self.model.state_dict().items():
            delta = 0
            for msg in messages:
                delta += layer_W - msg.source_client.model.state_dict()[layer_key]
            layer_W = layer_W + (self._gamma / len(messages)) * delta
