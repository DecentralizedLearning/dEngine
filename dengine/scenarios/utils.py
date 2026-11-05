from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from dengine.interfaces import ClientInterface


@contextmanager
def client_on_device_context(client: ClientInterface, device: torch.device):
    client.model.to(device)
    yield
    client.model.cpu()


@contextmanager
def cuda_stream_context(device: torch.device):
    if device.type != 'cuda':
        yield
        return

    current_stream = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(current_stream)
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        current_stream.wait_stream(stream)


@contextmanager
def cuda_client(client: ClientInterface, device: torch.device):
    with cuda_stream_context(device), client_on_device_context(client, device):
        yield


class SerialExecutor:
    def submit(self, fn, *args, **kwargs) -> Future:
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:
            f.set_exception(e)
        return f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def executor_context(num_workers: int):
    return SerialExecutor() if num_workers <= 1 else ThreadPoolExecutor(max_workers=num_workers)
