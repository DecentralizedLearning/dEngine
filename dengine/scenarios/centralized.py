from warnings import deprecated
from typing import Sequence

import torch

from dengine.scenarios.utils import client_on_device_context
from dengine.interfaces import MessageBase

from .scenario import AbstractClient, AbstractScenarioEngine
from .decorators import register_scenario, register_client


@register_client()
class CentralizedClient(AbstractClient):
    def aggregation(self, messages: Sequence[MessageBase]):
        assert len(messages) == 0


@register_scenario()
class CentralizedScenarioEngine(AbstractScenarioEngine[CentralizedClient]):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        super().__init__(*args, **kwargs, common_init=True)
        assert len(self.graph.nodes) == 1
        self._centralized_client = list(self.clients.values())[0]

    def run(self):
        with client_on_device_context(self._centralized_client, self._device):
            self._centralized_client.update(0)
            self._centralized_client.test(0, self.test_data)


@deprecated("This class has been refactored to CentralizedClient, please consider refactoring.")
@register_client()
class CentralizedPaiv(CentralizedClient):
    def __init__(self, *args, test_batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
