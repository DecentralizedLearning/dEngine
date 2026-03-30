from typing import List, Tuple, Dict
from omegaconf import OmegaConf
from pathlib import Path
from random import randint
from enum import Enum
from dataclasses import dataclass, field
from argparse import ArgumentParser

from dengine.config.builtins import get_config


class VerbosityLevel(Enum):
    silent = 0
    debug = 1
    info = 2


@dataclass
class SimulationArguments:
    gpus: List[int]
    torch_num_threads: int
    verbosity: VerbosityLevel
    seed: int
    resume_checkpoints: bool
    dataset_directory: Path
    sanity_check: bool
    output_directory: Path
    dump_stdout: bool
    configs: List[Path] = field(default_factory=list)


def cli_argument_parser(
    parser: ArgumentParser | None = None
) -> Tuple[SimulationArguments, Dict[str, str]]:
    if parser:
        parser_handler = parser.add_argument_group("Simulator Options")
    else:
        parser = ArgumentParser()
        parser_handler = parser
    parser_handler.add_argument(
        '--config',
        type=str,
        nargs='+',
        default=[],
        help='yaml configuration. Note: will override current parameters'
    )
    parser_handler.add_argument(
        '--resume_checkpoints',
        action='store_true',
        help='If enabled the models checkpoints will be loaded before starting the training'
    )
    parser_handler.add_argument(
        '--dump_stdout',
        action='store_true',
        help='Save logs to output_directory/logs/'
    )
    parser_handler.add_argument(
        '--sanity_check',
        action='store_true',
        help='Performs a sanity check of the configuration file. No experiment is run'
    )
    parser_handler.add_argument(
        '--gpu',
        type=str,
        default='0',
        help="GPU ID, -1 for CPU"
    )
    parser_handler.add_argument(
        '--torch_num_threads',
        type=int,
        help='Sets the number of threads used by torch for intraop parallelism on CPU',
        default=1
    )
    parser_handler.add_argument(
        '--verbosity',
        type=str,
        default=VerbosityLevel.silent.name,
        choices=[x.name for x in VerbosityLevel]
    )
    parser_handler.add_argument(
        '--dataset_directory',
        type=Path,
        default='./datasets',
    )
    parser_handler.add_argument(
        '--output_directory',
        type=Path,
        default='./logs',
    )
    parser_handler.add_argument('--seed', type=int, default=-1, help='random seed (default: random)')
    args, kwargs = parser.parse_known_args()
    overrides = dict(OmegaConf.from_dotlist(kwargs))

    gpu = [int(x) for x in args.gpu.split(',')]
    verbosity = VerbosityLevel[args.verbosity]
    seed = args.seed if args.seed >= 0 else randint(0, 424242)
    config_files = []

    for f in args.config:
        f = Path(f)
        if f.exists() and f.is_file():
            config_files.append(f)
            continue
        if (core_config := get_config(f.name)).exists() and core_config.is_file():
            config_files.append(core_config)
            continue
        raise ValueError(f'Not found: {f}')

    return SimulationArguments(
        configs=config_files,
        verbosity=verbosity,
        gpus=gpu,
        torch_num_threads=args.torch_num_threads,
        seed=seed,
        dataset_directory=args.dataset_directory,
        resume_checkpoints=args.resume_checkpoints,
        sanity_check=args.sanity_check,
        output_directory=Path(args.output_directory),
        dump_stdout=args.dump_stdout
    ), overrides
