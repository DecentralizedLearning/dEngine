from __future__ import annotations

from copy import deepcopy
from typing import Optional, Dict, List, Union
from pathlib import Path

from pydantic import BaseModel

from .constants import (
    METRICS_DIR_NAME,
    CHECKPOINT_DIR_NAME,
    GRAPH_DIR_NAME,
    PARTITIONS_DIR_NAME,
)


class DynamicModuleConfigBase(BaseModel):
    target: str
    arguments: Optional[Dict] = None


class ClientModuleConfig(DynamicModuleConfigBase):
    training_engine: DynamicModuleConfigBase
    local_model: DynamicModuleConfigBase


class DatasetModuleConfig(BaseModel):
    train: DynamicModuleConfigBase
    test: DynamicModuleConfigBase
    postprocessing: Optional[DynamicModuleConfigBase] = None


class ExperimentConfiguration(BaseModel):
    EXPERIMENT_VERSION: str = "1"

    target: Optional[str] = None
    experiments_directory_root: str
    name: str
    seed: int

    dataset: DatasetModuleConfig
    partitioning: Union[DynamicModuleConfigBase, List[DynamicModuleConfigBase]]
    graph: DynamicModuleConfigBase
    scenario: DynamicModuleConfigBase
    client: ClientModuleConfig
    callbacks: List[DynamicModuleConfigBase] = []

    @property
    def output_directory(self) -> Path:
        return Path(self.experiments_directory_root) / self.name

    @property
    def metrics_output_directory(self) -> Path:
        return self.output_directory / METRICS_DIR_NAME

    @property
    def checkpoint_output_directory(self) -> Path:
        return self.output_directory / CHECKPOINT_DIR_NAME

    @property
    def graph_output_directory(self) -> Path:
        return self.output_directory / GRAPH_DIR_NAME

    @property
    def partitions_output_directory(self) -> Path:
        return self.output_directory / PARTITIONS_DIR_NAME

    def migrate(self, **kwargs) -> ExperimentConfiguration:
        return self


class PaivModuleConfig(DynamicModuleConfigBase):
    training_engine: DynamicModuleConfigBase


class LegacyExperimentConfiguration(BaseModel):
    target: Optional[str] = None
    experiments_directory_root: str
    name: str
    seed: int

    dataset_train: DynamicModuleConfigBase
    dataset_test: DynamicModuleConfigBase
    dataset_postprocessing: Optional[DynamicModuleConfigBase] = None
    partitioning: Union[DynamicModuleConfigBase, List[DynamicModuleConfigBase]]
    graph: DynamicModuleConfigBase
    local_model: DynamicModuleConfigBase
    scenario: DynamicModuleConfigBase
    paiv: PaivModuleConfig
    callbacks: List[DynamicModuleConfigBase] = []

    @property
    def output_directory(self) -> Path:
        return Path(self.experiments_directory_root) / self.name

    @property
    def metrics_output_directory(self) -> Path:
        return self.output_directory / METRICS_DIR_NAME

    @property
    def checkpoint_output_directory(self) -> Path:
        return self.output_directory / CHECKPOINT_DIR_NAME

    @property
    def graph_output_directory(self) -> Path:
        return self.output_directory / GRAPH_DIR_NAME

    @property
    def partitions_output_directory(self) -> Path:
        return self.output_directory / PARTITIONS_DIR_NAME

    def migrate(self, **kwargs):
        experiment_cfg = deepcopy(self)
        if experiment_cfg.scenario.arguments:
            experiment_cfg.scenario.arguments['max_communication_rounds'] = experiment_cfg.scenario.arguments['communication_rounds']
            del experiment_cfg.scenario.arguments['communication_rounds']
        experiment_cfg.scenario.target = experiment_cfg.scenario.target.split(".")[-1]
        experiment_cfg.paiv.target = experiment_cfg.paiv.target.split(".")[-1]
        experiment_cfg.paiv.training_engine.target = experiment_cfg.paiv.training_engine.target.split(".")[-1]
        experiment_cfg.local_model.target = experiment_cfg.local_model.target.split(".")[-1]
        for cb in experiment_cfg.callbacks:
            cb.target = cb.target.split(".")[-1]

        if experiment_cfg.scenario.target == "CentralizedScenarioEngine":
            experiment_cfg.scenario.arguments = {}
        if experiment_cfg.scenario.target == "VanillaFederatedScenario":
            if experiment_cfg.scenario.arguments is None:
                experiment_cfg.scenario.arguments = {}
            experiment_cfg.scenario.arguments["use_weighted_avg"] = experiment_cfg.scenario.arguments.get("weighted_avg", True)
            del experiment_cfg.scenario.arguments["weight_avg"]
            experiment_cfg.scenario.arguments["common_init"] = experiment_cfg.scenario.arguments.get("common_init", True)

        paiv = experiment_cfg.paiv
        return ExperimentConfiguration(
            experiments_directory_root=experiment_cfg.experiments_directory_root,
            client=ClientModuleConfig(
                target=paiv.target,
                arguments=paiv.arguments,
                local_model=experiment_cfg.local_model,
                training_engine=experiment_cfg.paiv.training_engine
            ),
            callbacks=experiment_cfg.callbacks,
            dataset=DatasetModuleConfig(
                train=experiment_cfg.dataset_train,
                test=experiment_cfg.dataset_test,
                postprocessing=experiment_cfg.dataset_postprocessing
            ),
            graph=experiment_cfg.graph,
            name=experiment_cfg.name,
            partitioning=experiment_cfg.partitioning,
            scenario=experiment_cfg.scenario,
            seed=experiment_cfg.seed
        ).migrate(**kwargs)
