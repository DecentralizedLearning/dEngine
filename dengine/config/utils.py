import os
from datetime import datetime
import logging
import re
import ast
import yaml
import sys
import importlib
from pathlib import Path
from types import ModuleType
from typing import Optional, Type, TypeVar, List, Union, Tuple, Dict, Any

from omegaconf import OmegaConf
from pydantic.v1.utils import deep_update

import dengine

from .configuration import ExperimentConfiguration, DynamicModuleConfigBase

TT = TypeVar('TT')


def convert_to_nested_dict(
    plain_dict: Dict[str, Any]
) -> Dict[str, Any]:
    nested_dict = {}
    for key, value in plain_dict.items():
        keys = key.split('.')
        d = nested_dict
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return nested_dict


_ENV_RE = re.compile(r'\$(\w+)|\$\{(\w+)\}')


def _expand_env_in_str(s: str) -> Any:
    m = _ENV_RE.match(s)
    if not m:
        return s
    env_var_name = m.group(1) or m.group(2)
    env_value = os.environ.get(env_var_name, None)
    if not env_value:
        raise ValueError(f"❌ {env_var_name} is missing. Please define it via an environment variable.")
    casted_env_value = ast.literal_eval(env_value)
    return casted_env_value


def replace_env_variables(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: replace_env_variables(v) for k, v in data.items()}
    if isinstance(data, list):
        return [replace_env_variables(x) for x in data]
    if isinstance(data, tuple):
        return tuple(replace_env_variables(x) for x in data)
    if isinstance(data, set):
        return {replace_env_variables(x) for x in data}
    if isinstance(data, str):
        return _expand_env_in_str(data)
    return data


def load_experiment_from_yamls(
    files: List[Path],
    validator: Type[TT] = ExperimentConfiguration,
    overrides: Dict = {},
    **kwargs,
) -> TT:
    exp_name = []
    config = {}

    for f in files:
        with open(f, 'r') as content:
            logging.info(f'Overwriting configuration with: {f}')
            data = yaml.safe_load(content)
            data = replace_env_variables(data)
            config.update(data)
            name = data['name'] if 'name' in data else f.stem
            exp_name.append(name)

    if ("seed" in config) and ("seed" in kwargs):
        del kwargs["seed"]
    config = deep_update(config, convert_to_nested_dict(kwargs))
    config = OmegaConf.to_container(OmegaConf.merge(config, overrides))
    if not isinstance(config, Dict):
        raise ValueError(f"Bad configuration. Expected dict, found: {type(config)}")

    if "name" not in config:
        config["name"] = (
            ",".join(exp_name) +
            f',seed={config["seed"]}' +
            ': ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    return validator(**config)


def load_configuration_module(
    config: DynamicModuleConfigBase,
    reload=False,
    from_module: Optional[ModuleType] = None,
):
    if from_module is None:
        package = None
        module, cls = config.target.rsplit(".", 1)
        if module.startswith('.'):
            sys.path.append(os.getcwd())
            package = os.getcwd()
            module = module[1:]
        from_module = importlib.import_module(module, package=package)
    else:
        path_in_module = config.target.split('.')
        if len(path_in_module) == 0:
            cls = config.target
        else:
            cls = path_in_module[-1]
            module_path = [from_module.__name__] + path_in_module[:-1]
            from_module = importlib.import_module(
                ".".join(module_path),
                package=from_module.__package__
            )

    if reload:
        importlib.reload(from_module)

    cls = getattr(from_module, cls)
    return cls


def load_configuration_module_experimental_fallback(
    config: DynamicModuleConfigBase,
    reload=False,
    from_module: Optional[ModuleType] = None,
):
    try:
        cls = load_configuration_module(config, reload, from_module)
        return cls
    except (ModuleNotFoundError, ValueError):
        pass

    try:
        cls = load_configuration_module(config, reload, dengine)
        return cls
    except (ModuleNotFoundError, ValueError):
        pass

    try:
        cls = load_configuration_module(config, reload)
        return cls
    except (ModuleNotFoundError, ValueError):
        pass

    raise ModuleNotFoundError(f"No module named {config.target}")


T = TypeVar('T')


def instantiate_configuration_module(
    config: DynamicModuleConfigBase,
    reload: bool = False,
    from_module: Optional[ModuleType] = None,
    superclass: Union[
        Type[T],
        Tuple[Type[T], ...],
        List[Type[T]],
    ] = (object, type(lambda: None)),
    allow_experimental: bool = True,
    allowed_cls: Optional[Dict[str, Any]] = None,
    **kwargs
) -> T:
    if allowed_cls and config.target in allowed_cls:
        cls = allowed_cls[config.target]
    elif allow_experimental:
        cls = load_configuration_module_experimental_fallback(config, reload, from_module)
    else:
        cls = load_configuration_module(config, reload, from_module)

    if isinstance(superclass, list):
        superclass = tuple(superclass)
    if not isinstance(superclass, tuple):
        superclass = tuple([superclass])

    if isinstance(cls, Type):  # cls is a class
        cls_return_type = cls
    elif isinstance(cls, type(lambda: None)):  # cls is a function
        cls_return_type = cls.__annotations__['return']
    else:
        raise ValueError(f'Wrong type for {cls}')

    supported_superclasses = [x.__name__ for x in superclass]
    if (
        isinstance(cls_return_type, Type) and
        (not any([issubclass(cls_return_type, x) for x in superclass]))
    ):
        raise TypeError(
            f"The loaded class {cls.__name__} is not a subclass of {supported_superclasses}"
        )
    elif (not any([cls_return_type == x] for x in superclass)):
        raise TypeError(
            f"The loaded type {cls.__name__} is not a subclass of {supported_superclasses}"
        )

    config_args = config.arguments or {}
    return cls(**config_args, **kwargs)


def dump_experiment(exp: ExperimentConfiguration):
    raise NotImplementedError()
