from typing import Tuple
from pathlib import Path
import re
from argparse import ArgumentParser
import signal
import torch.multiprocessing as mp
import json

from tqdm import tqdm
from pydantic import ValidationError

from dengine.bin.args_parser import cli_argument_parser
from dengine.config.builtins import BUILTINS
from dengine.bin import SimulationArguments
from dengine.config import ExperimentConfiguration
from dengine.config.utils import convert_to_nested_dict
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import load_engine
from dengine.scenarios.decentralized import DecAvgClient
from dengine.scenarios.centralized import CentralizedClient


# ..... ..... ..... ..... ..... ..... ..... ..... #
# SCENARIOS
# ..... ..... ..... ..... ..... ..... ..... ..... #
FEDERATED_CONFIGS = [
    BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
    BUILTINS.CORE.GRAPH.STAR_51,
    BUILTINS.CORE.DATASETS.MNIST,
    Path("configs/core.yml")
]
DECENTRALIZED_BA_CONFIGS = [
    BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
    BUILTINS.CORE.GRAPH.BA_MEDIUM,
    BUILTINS.CORE.DATASETS.MNIST,
    Path("configs/core.yml")
]
DECENTRALIZED_ER_CONFIGS = [
    BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
    BUILTINS.CORE.GRAPH.ER_MEDIUM,
    BUILTINS.CORE.DATASETS.MNIST,
    Path("configs/core.yml")
]
CENTRALIZED_CONFIGS = [
    BUILTINS.CORE.GRAPH.CENTRALIZED,
    BUILTINS.CORE.SCENARIOS.CENTRALIZED,
    BUILTINS.CORE.DATASETS.MNIST,
    Path("configs/core.yml"),
    Path("configs/centralized_callbacks.yml")
]

CENTRALIZED_CONFIG_OVERRIDES = {
    "client.target": CentralizedClient.__name__,
    "client.training_engine.arguments.epochs": 1500,
}
DECENTRALIZED_CONFIG_OVERRIDES = {
    "client.target": DecAvgClient.__name__,
    # Aggregation and Scenario
    "scenario.arguments.common_init": True,
    "client.arguments.include_myself": True,
    "client.arguments.use_weighted_avg": True,
    "scenario.arguments.max_communication_rounds": 200,
    # Training Engine
    "client.training_engine.arguments.optimizer": "Adam",
    "client.training_engine.arguments.lr": 0.0003,
    "client.training_engine.arguments.adam_weight_decay": 0.001,
    "client.training_engine.arguments.scheduler": "cosine",
    "client.training_engine.arguments.patience": 5,
    "client.training_engine.arguments.epochs": 5,
    "client.training_engine.arguments.validation_batch_size": 32,
    "client.training_engine.arguments.training_batch_size": 32,
    "partitioning.arguments.validation_percentage": 0.1,
}


# ..... ..... ..... ..... ..... ..... ..... ..... #
# ARCHITECTURE
# ..... ..... ..... ..... ..... ..... ..... ..... #
TINY_CNN = {"client.local_model.target": "TinyCNN"}
MOBILE_VNET = {"client.local_model.target": "MobileNetV3Classifier"}
WIDE_RESNET = {
    "client.local_model.target": "WideResNetClassifier",
    "client.local_model.arguments.depth": 10,
    "client.local_model.arguments.dropout": 0.3,
    "client.local_model.arguments.widen_factor": 5,
}
VIT = {"client.local_model.target": "ViT"}


def load_centralized_cifar10(
    simulation_args: SimulationArguments,
    architecture,
    debug: bool = False
):
    debug_argument = {
        "client.arguments.debug_skip_training": True,
        "client.arguments.debug_skip_test": True
    } if debug else {}
    model_name = architecture['client.local_model.target']
    return [
        load_experiment_from_yamls(
            files=[*CENTRALIZED_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": f"cifar10,Centralized,{model_name}",
                **CENTRALIZED_CONFIG_OVERRIDES,
                **debug_argument,
                **architecture,
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        )
    ]


def load_cifar10_decentralized(
    simulation_args: SimulationArguments,
    config_overrides,
    architecture,
    debug: bool = False
):
    debug_argument = {
        "client.arguments.debug_skip_training": True,
        "client.arguments.debug_skip_test": True
    } if debug else {}

    model_name = architecture['client.local_model.target']
    return [
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_BA_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": f"cifar10,BA,DecAvg,{model_name}",
                "graph.arguments.seed": simulation_args.seed,
                **config_overrides,
                **architecture,
                **debug_argument
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*DECENTRALIZED_ER_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": f"cifar10,ER,DecAvg,{model_name}",
                "graph.arguments.seed": simulation_args.seed,
                **config_overrides,
                **architecture,
                **debug_argument
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[*FEDERATED_CONFIGS],
            overrides=convert_to_nested_dict({
                "name": f"cifar10,FedAvg,{model_name}",
                **config_overrides,
                **architecture,
                **debug_argument
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
    ]


# ..... ..... ..... ..... ..... ..... ..... ..... #
# MAIN
# ..... ..... ..... ..... ..... ..... ..... ..... #
def _worker_init():
    """Ignore SIGINT in workers — let the parent process handle shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_config(
    args_tuple: Tuple[SimulationArguments, ExperimentConfiguration]
):
    simulation_args, cfg = args_tuple
    try:
        loaded_engine = load_engine(simulation_args, cfg, verbose=False)
        loaded_engine.run()
    except Exception as e:
        raise RuntimeError(f"Config {cfg} failed: {e}") from e


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--concurrent-runs",
        type=int,
        default=1,
        help="Number of configurations to run in parallel (default: 1)."
    )
    parser.add_argument(
        "--experiment_name_filter",
        type=str,
        action="append",
        default=None,
        help="Filter experiments names with a regex."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Sets the argument client.arguments.debug_skip_training=true"
    )
    simulation_args, _ = cli_argument_parser(parser)
    extra_arguments = parser.parse_args()

    configurations = []
    try:
        for architecture_cfg in [TINY_CNN, MOBILE_VNET, WIDE_RESNET, VIT]:
            configurations += [
                *load_centralized_cifar10(simulation_args, architecture=architecture_cfg, debug=extra_arguments.debug),
                *load_cifar10_decentralized(simulation_args, debug=extra_arguments.debug, architecture=architecture_cfg, config_overrides=DECENTRALIZED_CONFIG_OVERRIDES),
            ]
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
        return

    if extra_arguments.experiment_name_filter:
        selected_experiments = set()
        for pattern in extra_arguments.experiment_name_filter:
            regex_pattern = re.compile(pattern)
            selected_experiments.update(
                [cfg.name for cfg in configurations if regex_pattern.search(cfg.name)]
            )
        configurations = [cfg for cfg in configurations if cfg.name in selected_experiments]
        print(f"Found {len(configurations)} configs:")
        print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        try:
            input("Type [ENTER] to continue...")
        except KeyboardInterrupt:
            return

    if simulation_args.sanity_check:
        json_outfname = "experiments_configs.json"
        print(f'🟢 Configurations are fine. Dumped the following configs to {json_outfname}.')
        if not extra_arguments.experiment_name_filter:
            print(f"Found {len(configurations)} configs:")
            print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        print("Ready to run...")
        all_configs = [cfg.model_dump() for cfg in configurations]
        with open(json_outfname, "w") as f:
            json.dump(all_configs, f, indent=2)
        return

    if len(configurations) == 1:
        try:
            loaded_engine = load_engine(simulation_args, configurations[0], verbose=False)
            loaded_engine.run()
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            raise RuntimeError(f"Config {configurations[0]} failed: {e}") from e

    pool_work_items = [(simulation_args, cfg) for cfg in configurations]
    try:
        with mp.Pool(processes=extra_arguments.concurrent_runs, initializer=_worker_init, maxtasksperchild=1) as pool:
            with tqdm(total=len(configurations), desc="Running configs", unit="cfg") as pbar:
                for _ in pool.imap_unordered(_run_config, pool_work_items):
                    pbar.update()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
