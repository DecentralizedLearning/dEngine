import sys
from pathlib import Path

from pydantic import ValidationError

from dengine import load_experiment_from_yamls, run_simulation
from dengine.bin.args_parser import cli_argument_parser


PROJECT_ROOT_PATH = Path(__file__).parent.parent
sys.path.extend([
    str(PROJECT_ROOT_PATH.absolute()),
])


if __name__ == "__main__":
    args, overrides = cli_argument_parser()
    try:
        cfg = load_experiment_from_yamls(
            args.configs,
            overrides=overrides,
            output_directory=args.output_directory,
            seed=args.seed,
        )
        run_simulation(args, cfg)
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
