from __future__ import annotations

from typing import List, Dict, Optional, Union, Tuple, Self
from glob import glob
from collections import defaultdict
from pathlib import Path
from npy_append_array import recover

import numpy as np

from dengine.config.constants import METRICS_DIR_NAME
from .metrics import precision, recall, f1


def find_metrics(experiment: Path) -> Dict[str, List[Path]]:
    confusion_matrix_path = experiment / METRICS_DIR_NAME
    confusion_matrices_path_str = glob(str(confusion_matrix_path) + '/**/*.npy', recursive=True)
    confusion_matrices_paths = [Path(n) for n in confusion_matrices_path_str]
    confusion_matrices_paths.sort(key=lambda x: int(x.parent.stem))

    metrics = defaultdict(list)
    for fpath in confusion_matrices_paths:
        metrics[fpath.name].append(fpath)
    return metrics


def find_confusion_matrices(experiment: Path) -> Dict[str, List[Path]]:
    return find_metrics(experiment)


def _normalize_confusion_matrix_rows(confusion_matrix: np.ndarray) -> np.ndarray:
    confusion_matrix = confusion_matrix.astype(float)
    confusion_matrix_sum = confusion_matrix.sum(axis=-1, keepdims=True)
    confusion_matrix[confusion_matrix == 0] = np.nan
    normalized = confusion_matrix / confusion_matrix_sum
    normalized[np.isnan(normalized)] = 0.
    return normalized


def _accuracy(
    rounds_confusion_matricies: np.ndarray,
    class_idxs: Optional[List[int]] = None
) -> np.ndarray:
    rounds, classes, _ = rounds_confusion_matricies.shape
    if class_idxs is None:
        class_idxs = list(range(classes))

    round_accuracy = []
    for cf in rounds_confusion_matricies:
        identity = np.identity(classes).astype(bool)
        true_positive = cf[class_idxs][identity[class_idxs]].sum()
        misclassified = cf[class_idxs][~identity[class_idxs]].sum()
        accuracy = true_positive / (true_positive + misclassified)
        round_accuracy.append(accuracy)
    return np.array(round_accuracy)


class ExperimentMetric:
    def __init__(
        self,
        confusion_matrix_npy_fpaths: Optional[List[Path]] = None,
        devices_confusion_matrices: Optional[np.ndarray] = None,
        description: Optional[str] = None,
        time_devices_confusion_matrices: Optional[np.ndarray] = None,
    ):
        time_data = time_devices_confusion_matrices

        if devices_confusion_matrices is not None:
            self._description = description
            data = devices_confusion_matrices
        elif confusion_matrix_npy_fpaths is not None:
            self._description = confusion_matrix_npy_fpaths[0].parent.parent.parent.name
            # If files appears to be truncated, this line fixes the problem
            [recover(cf_path) for cf_path in confusion_matrix_npy_fpaths]
            data = [np.load(cf_fpath) for cf_fpath in confusion_matrix_npy_fpaths]

            time_confusion_matrix_npy_fpaths = [
                cf_fpath.parent / f".{cf_fpath.stem}.time{cf_fpath.suffix}"
                for cf_fpath in confusion_matrix_npy_fpaths
            ]
            if all([p.exists() for p in time_confusion_matrix_npy_fpaths]):
                [recover(cf_path) for cf_path in time_confusion_matrix_npy_fpaths]
                time_data = [np.load(cf_fpath) for cf_fpath in time_confusion_matrix_npy_fpaths]
        else:
            raise ValueError('Neither partition nor experiment are specified')

        self.rounds = max([len(X) for X in data])
        self._data = np.full((len(data), self.rounds, *data[0].shape[1:]), np.nan)
        for i, X in enumerate(data):
            self._data[i, :len(X)] = X

        if time_data is not None:
            self._time_data = np.stack([X[:self.rounds] for X in time_data])
        else:
            self._time_data = None

    @property
    def data(self):
        return self._data

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def time_data(self):
        return self._time_data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    @property
    def epochs(self):
        return self.data.shape[1]

    def truncate_at_epoch_view(self, max_epoch: Union[int, Tuple, slice]) -> Self:
        if isinstance(max_epoch, Tuple):
            if self.rounds == 1:
                return self
            return self.__class__(
                devices_confusion_matrices=self.data[:, max_epoch[0]:max_epoch[1]],
                time_devices_confusion_matrices=self._time_data[:, max_epoch[0]:max_epoch[1]] if self._time_data is not None else None,
                description=self._description
            )
        return self.__class__(
            devices_confusion_matrices=self.data[:, :max_epoch],
            time_devices_confusion_matrices=self._time_data[:, :max_epoch] if self._time_data is not None else None,
            description=self._description
        )

    def epoch_view(self, epochs: List[int]) -> Self:
        return self.__class__(
            devices_confusion_matrices=self.data[:, epochs],
            time_devices_confusion_matrices=self._time_data[:, epochs] if self._time_data is not None else None,
            description=self._description
        )

    def device_view(self, devices: List) -> Self:
        return self.__class__(
            devices_confusion_matrices=self.data[devices],
            time_devices_confusion_matrices=self._time_data[devices] if self._time_data is not None else None
        )


class ExperimentConfusionMatrix(ExperimentMetric):
    def class_indexes(self) -> List[int]:
        classes_cardinality = self.data[0][0].shape[0]
        return list(range(classes_cardinality))

    def confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        if normalize:
            return _normalize_confusion_matrix_rows(self.data)
        return self.data

    def accuracy(self, idx_labels: Optional[List[int]] = None) -> np.ndarray:
        return np.stack([
            _accuracy(X, idx_labels) for X in self.data
        ])

    def mean_accuracy(self) -> np.ndarray:
        return np.mean(self.accuracy(), axis=0)

    def _ith_label_metric_over_time(
        self,
        rounds_confusion_matricies: np.ndarray,
        i: int,
        metric,
    ) -> np.ndarray:
        round_accuracy = []
        for cf in rounds_confusion_matricies:
            row_selector = np.zeros_like(cf, dtype=bool)
            row_selector[i] = True
            row_selector[i][i] = False

            col_selector = np.zeros_like(cf, dtype=bool)
            col_selector[:, i] = True
            col_selector[i][i] = False

            TP = cf[i][i]
            FP = cf[col_selector].sum()
            FN = cf[row_selector].sum()
            TN = cf[~(col_selector | row_selector.T)].sum() - TP

            round_accuracy.append(
                metric(TP, FP, FN, TN)
            )
        return np.array(round_accuracy)

    def recall(self, idx_label: int):
        res = []
        for X in self.data:
            res.append(
                self._ith_label_metric_over_time(X, idx_label, recall)
            )
        return np.stack(res)

    def precision(self, idx_label: int):
        res = []
        for X in self.data:
            res.append(
                self._ith_label_metric_over_time(X, idx_label, precision)
            )
        return np.stack(res)

    def f1(self, idx_label: int):
        res = []
        for X in self.data:
            res.append(
                self._ith_label_metric_over_time(X, idx_label, f1)
            )
        return np.stack(res)

    def __sub__(self, other: ExperimentConfusionMatrix):
        return ExperimentConfusionMatrix(
            devices_confusion_matrices=self.data - other.data
        )


class ExperimentConfusionMatrixDelta(ExperimentConfusionMatrix):
    def __init__(self, a: ExperimentConfusionMatrix, b: ExperimentConfusionMatrix):
        super().__init__(devices_confusion_matrices=a.data)
        self._a = a
        self._b = b

    def truncate_at_epoch_view(self, *args, **kwargs) -> ExperimentConfusionMatrixDelta:
        return ExperimentConfusionMatrixDelta(
            self._a.truncate_at_epoch_view(*args, **kwargs),
            self._b.truncate_at_epoch_view(*args, **kwargs),
        )

    def epoch_view(self, *args, **kwargs) -> ExperimentConfusionMatrixDelta:
        return ExperimentConfusionMatrixDelta(
            self._a.epoch_view(*args, **kwargs),
            self._b.epoch_view(*args, **kwargs),
        )

    def device_view(self, *args, **kwargs) -> ExperimentConfusionMatrixDelta:
        return ExperimentConfusionMatrixDelta(
            self._a.device_view(*args, **kwargs),
            self._b.device_view(*args, **kwargs),
        )

    def confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        return (
            self._a.confusion_matrix(normalize) -
            self._b.confusion_matrix(normalize)
        )
