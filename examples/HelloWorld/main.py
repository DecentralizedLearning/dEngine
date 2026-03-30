from pathlib import Path

from pydantic import ValidationError

import dengine
from dengine import load_experiment_from_yamls
from dengine.config.utils import convert_to_nested_dict
from dengine.bin.args_parser import cli_argument_parser
from dengine.bin.simulation import run_simulation, SimulationArguments, VerbosityLevel

from src import engine


def cli_main():
    """Run with: `python main.py --config config.yml --output_directory debug`
    """
    args, overrides = cli_argument_parser()
    try:
        cfg = load_experiment_from_yamls(
            args.configs,
            overrides=overrides,
            experiments_directory_root=str(args.output_directory),
            seed=args.seed,
        )
        run_simulation(args, cfg, engine)
    except KeyboardInterrupt:
        print('Exit')
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")


def hardcoded_main():
    """Run with: `python main.py --output_directory debug`
    """
    cli_arguments = SimulationArguments(
        gpus=[0],
        dataset_directory=Path('datasets/'),
        dump_stdout=False,
        output_directory=Path('logs/'),
        resume_checkpoints=False,
        sanity_check=False,
        seed=123,
        torch_num_threads=1,
        verbosity=VerbosityLevel.info
    )
    try:
        simulation_config = load_experiment_from_yamls(
            [
                dengine.BUILTINS.CORE.GRAPH.BA_SMALL,
                dengine.BUILTINS.CORE.PARTITIONING.IID,
                dengine.BUILTINS.CORE.DATASETS.MNIST,
                dengine.BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
            ],
            experiments_directory_root=str(cli_arguments.output_directory),
            overrides=convert_to_nested_dict({
                "client.training_engine.arguments.epochs": 10
            }),
            seed=cli_arguments.seed,
        )
        run_simulation(cli_arguments, simulation_config, engine)
    except KeyboardInterrupt:
        print('Exit')
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")


if __name__ == "__main__":
    cli_main()
