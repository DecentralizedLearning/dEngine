from __future__ import annotations

from pathlib import Path
from functools import partial

from dengine import dataset
from dengine.config.configuration import LegacyExperimentConfiguration
from dengine import graph
from dengine import AbstractClient
from dengine.graph import Graph
from dengine.dataset import SupervisedDataset
from dengine.config import instantiate_configuration_module, load_experiment_from_yamls
from dengine import partitioning
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.models.utils import load_last_checkpoint
from dengine.scenarios import ScenarioEngineInterface
from dengine.callbacks import callback_factory

from dengine.scenarios.decorators import BUILTIN_SCENARIOS
from dengine.dataset.decorators import BUILTIN_DATASETS
from dengine.graph.decorators import BUILTIN_GRAPHS
from dengine.partitioning.decorators import BUILTIN_PARTITIONING

from .partitioning import ExperimentPartitions


class Experiment:
    def __init__(
        self,
        experiment_root_path: Path,
        dataset_root_path: Path
    ):
        self.experiment_cfg = load_experiment_from_yamls(
            [experiment_root_path / "config.yaml"],
            output_directory=experiment_root_path.parent,
            name=experiment_root_path.name
        )
        self.initialize_config(experiment_root_path, dataset_root_path)

    def initialize_config(self, experiment_root_path: Path, dataset_root_path: Path):
        self.experiment_cfg.callbacks = []

        self.network_graph = instantiate_configuration_module(
            self.experiment_cfg.graph,
            from_module=graph,
            superclass=Graph,
            experiment_cfg=self.experiment_cfg,
            allowed_cls=BUILTIN_GRAPHS
        )

        self.dataset_train = instantiate_configuration_module(
            self.experiment_cfg.dataset.train,
            from_module=dataset,
            output_path=dataset_root_path,
            experiment_cfg=self.experiment_cfg,
            allowed_cls=BUILTIN_DATASETS
        )
        self.dataset_test = instantiate_configuration_module(
            self.experiment_cfg.dataset.test,
            from_module=dataset,
            output_path=dataset_root_path,
            experiment_cfg=self.experiment_cfg,
            allowed_cls=BUILTIN_DATASETS
        )
        if self.experiment_cfg.dataset.postprocessing:
            self.dataset_train, self.dataset_test = instantiate_configuration_module(
                self.experiment_cfg.dataset.postprocessing,
                from_module=dataset,
                superclass=SupervisedDataset,
                dataset=self.dataset_train,
                test=self.dataset_test,
            )

        self.partitions = ExperimentPartitions(experiment_root_path)

        self._partitions = None
        if not isinstance(self.experiment_cfg.partitioning, list):
            self.experiment_cfg.partitioning = [self.experiment_cfg.partitioning]

        for i, partitioning_cfg in enumerate(self.experiment_cfg.partitioning):
            self._partitions = instantiate_configuration_module(
                partitioning_cfg,
                from_module=partitioning,
                allowed_cls=BUILTIN_PARTITIONING,
                superclass=TYPE_DATASET_PARTITIONING,
                dataset=self.dataset_train,
                test=self.dataset_test,
                partitions=self._partitions,
                graph=self.network_graph,
            )
        assert self._partitions is not None

        _callback_factory = partial(
            callback_factory,
            configuration=self.experiment_cfg.callbacks,
            experiment_configuration=self.experiment_cfg,
            test_data=self.dataset_test
        )
        self.training_engine: ScenarioEngineInterface[AbstractClient] = instantiate_configuration_module(
            self.experiment_cfg.scenario,
            allowed_cls=BUILTIN_SCENARIOS,
            callback_factory=_callback_factory,
            superclass=ScenarioEngineInterface,
            graph=self.network_graph,
            training_data=self.dataset_train,
            data_partitions=self._partitions,
            test_data=self.dataset_test,
            client_configuration=self.experiment_cfg.client
        )
        load_last_checkpoint(list(self.training_engine.clients.values()), experiment_root_path / "checkpoints/")


class LegacyExperiment(Experiment):
    def __init__(
        self,
        experiment_root_path: Path,
        dataset_root_path: Path,
        **kwargs
    ):
        self.experiment_cfg = load_experiment_from_yamls(
            [experiment_root_path / "config.yaml"],
            experiments_directory_root=str(experiment_root_path.parent.absolute()),
            name=experiment_root_path.name,
            validator=LegacyExperimentConfiguration
        ).migrate(**kwargs)
        self.initialize_config(experiment_root_path, dataset_root_path)
