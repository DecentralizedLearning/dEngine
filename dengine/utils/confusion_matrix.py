from typing import Optional, List

import torch
import numpy as np
from torch import Tensor
from sklearn.metrics import confusion_matrix

from dengine.dataset import SupervisedDataset

from .utils import softmax_with_argmax


def confusion_matrix_from_model_output(
    Y_hat: Tensor,
    dataset: SupervisedDataset,
    idxs: Optional[Tensor] = None,
    labels: Optional[List[int]] = None
) -> np.ndarray:
    if labels is None:
        labels = torch.unique(dataset.targets)

    if idxs is None:
        Y_ground_truth = dataset.targets
    else:
        Y_ground_truth = dataset.targets[idxs]
    Y_ground_truth = Y_ground_truth.cpu().numpy()

    Y_prediction = softmax_with_argmax(Y_hat).cpu().numpy()
    conf_matrix = confusion_matrix(
        Y_ground_truth,
        Y_prediction,
        labels=labels
    )
    return np.expand_dims(conf_matrix, axis=0)
