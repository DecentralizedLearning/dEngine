from pathlib import Path
from functools import partial
from typing import Tuple
import yaml
import logging
import sys
import os
import importlib


import torch

from dengine.config import (
    instantiate_configuration_module,
    ExperimentConfiguration
)
from dengine import dataset

from dengine.dataset.decorators import BUILTIN_DATASETS
from dengine.graph.decorators import BUILTIN_GRAPHS
from dengine.partitioning.decorators import BUILTIN_PARTITIONING

from dengine.dataset import SupervisedDataset
from dengine import partitioning
from dengine.partitioning import dump_partition
from dengine.interfaces import TYPE_DATASET_PARTITIONING
from dengine import graph
from dengine.utils.utils import configure_logger, seed_everything
from dengine.graph import Graph
from dengine.callbacks import callback_factory
from dengine.scenarios.decorators import BUILTIN_SCENARIOS
from dengine.scenarios.event_api.scenario import ScenarioEventEngine
from dengine.scenarios import ScenarioEngineInterface
from dengine.models.utils import load_last_checkpoint
from dengine.partitioning.utils import partitions_report
from dengine.interfaces import GenericClient

from .args_parser import SimulationArguments, VerbosityLevel


def load_engine(
    args: SimulationArguments,
    experiment_cfg: ExperimentConfiguration,
    engine: ScenarioEventEngine[GenericClient] | None = None,
    verbose: bool = True
):
    torch.set_num_threads(args.torch_num_threads)

    # 1. Initialization and global configurations
    seed_everything(experiment_cfg.seed)

    experiment_cfg.output_directory.mkdir(parents=True, exist_ok=True)
    with open(experiment_cfg.output_directory / 'config.yaml', 'w') as f:
        d = experiment_cfg.model_dump()
        yaml.safe_dump(d, f)

    if args.dump_stdout:
        log_path = experiment_cfg.output_directory / 'logs/'
        log_path.mkdir(parents=True, exist_ok=True)
        logfile_path = log_path / 'run_experiment.md'

        configure_logger(args.verbosity, logfile_path)
    else:
        configure_logger(args.verbosity)

    # 2. Datasets
    dataset_train = instantiate_configuration_module(
        experiment_cfg.dataset.train,
        from_module=dataset,
        superclass=[SupervisedDataset],
        output_path=args.dataset_directory,
        experiment_cfg=experiment_cfg,
        allowed_cls=BUILTIN_DATASETS
    )
    dataset_test = instantiate_configuration_module(
        experiment_cfg.dataset.test,
        from_module=dataset,
        superclass=[SupervisedDataset],
        output_path=args.dataset_directory,
        experiment_cfg=experiment_cfg,
        allowed_cls=BUILTIN_DATASETS
    )

    # 3. Network topology
    network_graph = instantiate_configuration_module(
        experiment_cfg.graph,
        from_module=graph,
        superclass=Graph,
        experiment_cfg=experiment_cfg,
        allowed_cls=BUILTIN_GRAPHS
    )
    network_graph.dump()

    # 4. Data partitioning
    partitions = None
    if not isinstance(experiment_cfg.partitioning, list):
        experiment_cfg.partitioning = [experiment_cfg.partitioning]

    for i, partitioning_cfg in enumerate(experiment_cfg.partitioning):
        partitions = instantiate_configuration_module(
            partitioning_cfg,
            from_module=partitioning,
            superclass=TYPE_DATASET_PARTITIONING,
            allowed_cls=BUILTIN_PARTITIONING,
            dataset=dataset_train,
            test=dataset_test,
            partitions=partitions,
            graph=network_graph,
        )
        if verbose:
            logging.info(
                f"# {partitioning_cfg.target} \n"
                f"{partitions_report(dataset_train, partitions)}"
            )
        dump_partition(
            experiment_cfg.partitions_output_directory / f'{i}.{partitioning_cfg.target}.csv',
            dataset_train,
            partitions
        )
    assert partitions is not None
    dump_partition(
        experiment_cfg.partitions_output_directory / 'train_partitions.csv',
        dataset_train,
        partitions
    )

    # 6. Dataset postprocessing (required to remap targets in some experiments)
    if experiment_cfg.dataset.postprocessing:
        dataset_train, dataset_test = instantiate_configuration_module(
            experiment_cfg.dataset.postprocessing,
            from_module=dataset,
            superclass=Tuple[
                SupervisedDataset,
                SupervisedDataset
            ],
            dataset=dataset_train,
            test=dataset_test,
        )

    # 7. Scenario configuration
    _callback_factory = partial(
        callback_factory,
        configuration=experiment_cfg.callbacks,
        experiment_configuration=experiment_cfg,
        test_data=dataset_test
    )
    if engine is None:
        training_engine = instantiate_configuration_module(
            experiment_cfg.scenario,
            allowed_cls=BUILTIN_SCENARIOS,
            callback_factory=_callback_factory,
            superclass=ScenarioEngineInterface,
            graph=network_graph,
            training_data=dataset_train,
            data_partitions=partitions,
            test_data=dataset_test,
            client_configuration=experiment_cfg.client
        )
    else:
        engine_extra_args = experiment_cfg.scenario.arguments or {}
        engine.load(
            graph=network_graph,
            training_data=dataset_train,
            callback_factory=_callback_factory,
            data_partitions=partitions,
            test_data=dataset_test,
            client_configuration=experiment_cfg.client,
            **engine_extra_args
        )
        training_engine = engine

    if args.resume_checkpoints:
        load_last_checkpoint(
            list(training_engine.clients.values()),
            Path(experiment_cfg.checkpoint_output_directory)
        )

    return training_engine


def run_simulation(
    args: SimulationArguments,
    experiment_cfg: ExperimentConfiguration,
    engine: ScenarioEventEngine | None = None,
    verbose: bool = True
):
    training_engine = load_engine(args, experiment_cfg, engine, verbose)
    if args.sanity_check:
        if args.verbosity == VerbosityLevel.debug:
            print(experiment_cfg.model_dump_json(indent=2))
        print(f'🟢 Configuration {experiment_cfg.name} is fine, ready to run...')
        return

    if verbose:
        logging.info("<br><br><br>\n")
    try:
        training_engine.run()
    except KeyboardInterrupt:
        print('Exit')


def try_load_pwd__init__():
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    package_name = os.path.basename(cwd)

    if os.path.exists(os.path.join(cwd, "__init__.py")):
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        importlib.import_module(package_name)


def _cli():
    from pydantic import ValidationError
    from dengine import load_experiment_from_yamls
    from dengine.bin.args_parser import cli_argument_parser
    try_load_pwd__init__()

    args, overrides = cli_argument_parser()
    try:
        cfg = load_experiment_from_yamls(
            args.configs,
            overrides=overrides,
            experiments_directory_root=str(args.output_directory),
            seed=args.seed,
        )
        run_simulation(args, cfg)
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
