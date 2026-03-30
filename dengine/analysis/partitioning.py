from __future__ import annotations

from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from dengine.config.constants import TRAIN_PARTITIONS_NAME


class ExperimentPartitions:
    def __init__(
        self,
        experiment: Path,
        csv_relative_path: str = TRAIN_PARTITIONS_NAME
    ):
        train_partitions_csv = experiment / csv_relative_path
        self._partitions = pd.read_csv(train_partitions_csv)
        self._partitions = self._partitions.set_index('nodeid')
        self._partitions.sort_index(inplace=True)

    def devices(self) -> List[int]:
        return list(self._partitions.index.values)

    def targets(self) -> np.ndarray:
        return self._partitions.columns.values

    def train(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is not None:
            return self._partitions.iloc[idx].to_numpy()

        return self._partitions.sum().to_numpy()

    def train_df(self) -> pd.DataFrame:
        return self._partitions
