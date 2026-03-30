from typing import List, Tuple
import logging
from dataclasses import dataclass
from copy import deepcopy
import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from queue import PriorityQueue
from dengine.scenarios import (
    DecentralizedScenarioEngineBase,
    AbstractClient,
    ServerMockClient,
)
from dengine.interfaces import MessageBase
from dengine.utils.utils import model_on_device_context, assert_no_nans
from dengine.training_strategies.local_update_strategy import TrainingEngine, training_step_output
from dengine.training_strategies.decorators import register_local_training
from dengine.scenarios.decorators import register_scenario, register_client


def zeros_like_module(module: nn.Module):
    return [
        torch.zeros_like(param)
        for param in module.parameters()
    ]


def load_parameters_inplace(
    module: nn.Module,
    parameters: List[Parameter]
):
    state_dict = module.state_dict()
    for i, (layer_name, _) in enumerate(module.named_parameters()):
        state_dict[layer_name] = parameters[i]
    module.load_state_dict(state_dict)


@dataclass
class ScaffoldClientUpdateMessage(MessageBase):
    """
    Represents the message sent from the server to the client

    This message includes:
    - `x`: The client's local model parameters.
    - `c`: The client's control variate, which estimates the update direction to correct for client drift.

    This corresponds to the communication step outlined at line 5 of the original SCAFFOLD paper: https://arxiv.org/abs/1910.06378
    """
    x: List[Tensor]
    c: List[Tensor]
    ci: List[Tensor]


@dataclass
class ScaffoldServerUpdateMessage(MessageBase):
    """
    Represents the message sent from the client to the server

    This message includes:
    - `delta_x`: The difference between the global model parameters and the client's model parameters.
    - `delta_c`: The adjustment to the client's control variate based on the aggregated updates from all clients.

    These updates help align the client's model and control variate with the global model, mitigating issues arising from data heterogeneity.

    This corresponds to the communication step outlined at line 13 of the original SCAFFOLD paper: https://arxiv.org/abs/1910.06378
    """
    delta_y: List[Tensor]
    delta_c: List[Tensor]


@register_local_training()
class ScaffoldLocalUpdate(TrainingEngine):
    def __init__(
        self,
        *args,
        **kwargs,

    ):
        super().__init__(*args, **kwargs)
        self._c = None
        self._ci = None

    @property
    def c(self):
        assert self._c is not None
        return self._c

    @property
    def ci(self):
        assert self._ci is not None
        return self._ci

    @c.setter
    def c(self, value: List[Tensor]):
        self._c = value

    @ci.setter
    def ci(self, value: List[Tensor]):
        self._ci = value

    def training_step(self, net: nn.Module, optimizer: Optimizer, images: Tensor, labels: Tensor) -> training_step_output:
        net.train()
        device = next(net.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        assert_no_nans(output)

        erm = self._loss_func(output, labels)
        loss = torch.nanmean(erm, dim=0)

        optimizer.zero_grad()
        loss.backward()

        # Line 9-10 of Algorithm 1
        delta_c = [self.c[j] - self.ci[j] for j in range(len(self.c))]
        for param, c_d in zip(net.parameters(), delta_c):
            param.grad += c_d.data.to(param.device)

        optimizer.step()
        return training_step_output(output=output, reduced_loss=loss.item())


@register_client()
class ScaffoldClient(AbstractClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self._local_strategy, ScaffoldLocalUpdate)
        self._local_strategy: ScaffoldLocalUpdate
        self.c = zeros_like_module(self.model)
        self.ci = zeros_like_module(self.model)
        self._local_strategy.c = self.c
        self._local_strategy.ci = self.ci

    @property
    def msg_buffer(self) -> PriorityQueue[ScaffoldClientUpdateMessage]:
        return self._msg_buffer  # type: ignore

    def aggregation(self, messages: List[ScaffoldClientUpdateMessage]):
        if len(messages) == 0:
            return
        assert len(messages) == 1
        assert isinstance(messages[0], ScaffoldClientUpdateMessage)

        self.c = messages[0].c
        self.ci = messages[0].ci
        logging.info(f"Loaded model from {messages[0].source_client.id} at time {messages[0].time}")
        load_parameters_inplace(self.model, messages[0].x)


@register_scenario()
class FederatedScaffoldScenario(DecentralizedScenarioEngineBase):
    def __init__(
        self,
        *args,
        global_step_size=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        for i in self.get_active_clients():
            assert isinstance(i, ScaffoldClient)
        self._server_mock_client = ServerMockClient()

        self._eta_g = global_step_size

        self._global_model = deepcopy(self.clients['0'].model)
        self._num_layers = len(list(self._global_model.parameters()))
        self._c = zeros_like_module(self._global_model)
        self._N = len(self.clients)

    def get_active_clients(self) -> List[ScaffoldClient]:
        return super().get_active_clients()  # type: ignore

    def run(self):
        for round in range(self._max_communication_rounds):
            logging.info(f"# Starting communication round **{round}**")
            round_msgs = []
            round_ci = []
            for p in self.get_active_clients():
                msg, ci = self._client_step(round, p)
                round_msgs.append(msg)
                round_ci.append(ci)
            self.on_communication_round_end(round, round_msgs, round_ci)

    def _client_step(
        self,
        communication_round: int,
        client: ScaffoldClient
    ) -> Tuple[ScaffoldServerUpdateMessage, List[Tensor]]:
        logging.info(
            f"**[Client-{client.UUID}][CommunicationRound-{communication_round}]** \n"
            "```"
        )
        with model_on_device_context(client.model, self._device):
            client.execute_local_train_strategy(communication_round)
            logging.info("```\n```")
            client.test(communication_round, self.test_data)
            logging.info("```")
        return self.synchronize_client(communication_round, client)

    @torch.no_grad()
    def synchronize_client(
        self,
        communication_round: int,
        client: ScaffoldClient
    ) -> Tuple[ScaffoldServerUpdateMessage, List[Tensor]]:
        """Implements line 12-13-14 of Algorithm 1"""
        """Note: here the index i referes to the layer index instead of the client ID"""
        K = client._local_strategy._local_epochs
        x = list(
            deepcopy(self._global_model).parameters()
        )
        y = list(
            deepcopy(client.model).parameters()
        )

        c_plus = zeros_like_module(self._global_model)
        delta_c = zeros_like_module(self._global_model)
        delta_y = zeros_like_module(self._global_model)
        for i in range(self._num_layers):
            yi = y[i].to(x[i].device)
            c_plus[i] = client.ci[i] - self._c[i] + self._eta_g * (x[i] - yi) / K
            delta_y[i] = yi - x[i]
            delta_c[i] = c_plus[i] - client.ci[i]

        msg = ScaffoldServerUpdateMessage(
            time=communication_round,
            source_client=client,
            delta_y=delta_y,
            delta_c=delta_c
        )
        return msg, c_plus

    @torch.no_grad()
    def on_communication_round_end(
        self,
        round_idx: int,
        messages: List[ScaffoldServerUpdateMessage],
        round_ci: List[List[Tensor]]
    ):
        super().on_communication_round_end(round_idx)
        logging.info(
            "``` \n"
            f"Merging models coming from: {[p.UUID for p in self.get_active_clients()]} \n"
            "```"
        )

        clients = self.get_active_clients()
        S = len(clients)

        # Line 16-17 of Algorithm 1
        delta_x = zeros_like_module(self._global_model)
        delta_c = zeros_like_module(self._global_model)
        for p, msg in zip(clients, messages):
            for i in range(self._num_layers):
                delta_x[i] += msg.delta_y[i]
                delta_c[i] += msg.delta_c[i]
        delta_x = [layer / S for layer in delta_x]
        delta_c = [layer / S for layer in delta_c]

        x = list(deepcopy(self._global_model).parameters())
        c = deepcopy(self._c)
        for i in range(self._num_layers):
            x[i] = x[i] + (self._eta_g * delta_x[i])
            c[i] = c[i] + (S / self._N) * delta_c[i]

        load_parameters_inplace(self._global_model, x)

        # Line 5 of Algorithm 1
        for p, ci in zip(self.get_active_clients(), round_ci):
            msg = ScaffoldClientUpdateMessage(
                time=round_idx,
                source_client=self._server_mock_client,
                x=x,
                c=c,
                ci=ci
            )
            p.msg_buffer.put(msg)  # type: ignore
