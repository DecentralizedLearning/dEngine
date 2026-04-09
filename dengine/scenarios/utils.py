from contextlib import contextmanager

import torch

from dengine.interfaces import ClientInterface


@contextmanager
def client_on_device_context(client: ClientInterface, device: torch.device):
    client.model.to(device)
    yield
    client.model.cpu()
