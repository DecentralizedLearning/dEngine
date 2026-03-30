import os
from pathlib import Path
import random
from collections import defaultdict
from typing import Sequence, Optional, OrderedDict, Dict
from contextlib import contextmanager
import logging

import torch
from torch import Tensor, device
from torch.utils.data import DataLoader, Subset
from torch.nn import Module
import torch.nn as nn
import numpy as np
import pandas as pd

from dengine.bin.args_parser import VerbosityLevel


HLINE = "-" * 10


def still_gaining(train_loss):
    print("still_gaining:", train_loss)
    if len(train_loss) <= 1 or train_loss[-1] < train_loss[-2]:
        return True

    return False


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def softmax_with_argmax(log_probs: Tensor) -> Tensor:
    y_prob = nn.Softmax(dim=1)(log_probs)
    return y_prob.argmax(1)


def get_device(net: Module) -> device:
    return next(net.parameters()).device


@torch.no_grad()
def get_output_in_production(
    net: Module,
    dataset,
    idxs: Optional[Sequence[int] | Tensor],
    batch_size: int
) -> Tensor:
    net.eval()
    device = get_device(net)
    net_output = []
    if idxs is not None:
        dataset = Subset(dataset, idxs)
    loader = DataLoader(dataset, batch_size)

    for images, _ in loader:
        device_images = images.to(device)
        log_probs = net(device_images).cpu()
        net_output.extend(log_probs)
    return torch.stack(net_output)


def configure_logger(
    verbosity: VerbosityLevel,
    outfile_path: Optional[Path] = None
):
    if verbosity == VerbosityLevel.debug:
        format_str = '[%(asctime)s][%(module)s][%(name)s][%(filename)s:%(lineno)d]\n%(message)s'
        level = logging.INFO
    elif verbosity == VerbosityLevel.silent:
        level = logging.ERROR
        format_str = '%(message)s'
    else:
        level = logging.INFO
        format_str = '%(message)s'

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers

    formatter = logging.Formatter(format_str)

    # Always log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally log to file
    if outfile_path is not None:
        file_handler = logging.FileHandler(outfile_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


@contextmanager
def model_on_device_context(model: Module, device: torch.device):
    model.to(device)
    yield
    model.cpu()


@contextmanager
def state_dicts_on_same_device_context(
    state_dict: Dict[int, OrderedDict[str, Tensor]],
    model: Module
):
    device_state_dicts = defaultdict(dict)

    device = next(model.parameters()).device
    for id, model_parameters in state_dict.items():
        for key, layer in model_parameters.items():
            device_state_dicts[id][key] = layer.to(device)
    yield device_state_dicts
    for id, model_parameters in state_dict.items():
        for key, layer in model_parameters.items():
            device_state_dicts[id][key] = layer.cpu()


def tensor_report(X: Tensor, title: Optional[str] = None) -> str:
    unique_labels, count = torch.unique(X, return_counts=True)
    percentages = count / len(X)
    df = pd.DataFrame({
        "targets": unique_labels,
        "count": count,
        "percentages": percentages.round(decimals=2)
    })
    df['count'] = df['count'].astype(str)
    df['targets'] = df['targets'].astype(int)
    df = df.T
    df.columns = ['' for _ in df.columns]
    formatted_df = df.to_string(header="")
    header = f'{title} (**total samples: {len(X)}**) \n' if title else "\n"
    return (
        f'{header}```\n'
        f'{formatted_df} \n'
        '```'
    )


def assert_no_nans(X: Tensor):
    is_nan_mask = torch.any(torch.isnan(X))
    torch._assert(
        torch.logical_not(is_nan_mask),
        'output in train() is NaN'
    )


class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(self.file_path, 'a')

    def write(self, message):
        if message.strip():
            self.file.write(message.strip() + '\n')

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


@contextmanager
def markdown_codeblock(disabled: bool = False):
    if not disabled:
        logging.info("```")
        yield
        logging.info("```")
    else:
        yield
