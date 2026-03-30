from typing import List, Dict
from glob import glob
from pathlib import Path
from collections import defaultdict
from mako.template import Template

MAKO_TEMPLATE = "configs/builtins.py.mako"
BUILTIN_OUTPATH = "dengine/config/builtins.py"


def make_tree():
    return defaultdict(make_tree)


def tree2dict(obj):
    if isinstance(obj, defaultdict):
        return dict({k: tree2dict(v) for k, v in obj.items()})
    return obj


def set_nested(tree: Dict, path_to_the_leaf: List[str], value):
    curr_node = tree
    leaf = path_to_the_leaf[-1]
    for next_node in path_to_the_leaf[:-1]:
        curr_node = curr_node[next_node]
    curr_node[leaf] = value


def build_namespace(glob_pattern: str) -> Dict:
    root = make_tree()
    configs_fpaths = glob(glob_pattern, recursive=True)

    for cfg in configs_fpaths:
        pathlist = cfg.split('/')
        cfg_name = pathlist[-1]
        nesting = [dirname.upper() for dirname in pathlist[1:]]
        nesting[-1] = nesting[-1].split('.')[0]  # remove extension
        set_nested(root, nesting, (cfg_name, cfg))
    return tree2dict(root)


def main():
    tree = build_namespace("configs/**/*.yml")
    tmpl = Template(Path(MAKO_TEMPLATE).read_text())
    code = tmpl.render(ns_code=tree)
    Path(BUILTIN_OUTPATH).write_text(code)
    print("Wrote builtins.py")


if __name__ == "__main__":
    main()
