import logging
from typing import Optional
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

from dengine.config import ExperimentConfiguration

from .graph import Graph
from .decorators import register_graph


def _nx_dump(
    graph: nx.Graph,
    plot_outpath: Path,
    edgelist_outpath: Path
):
    plot_outpath.parent.mkdir(parents=True, exist_ok=True)
    edgelist_outpath.parent.mkdir(parents=True, exist_ok=True)

    weights = nx.get_edge_attributes(graph, 'weight').values()

    f = plt.figure()
    nx.draw(
        graph,
        pos=nx.spring_layout(graph),
        with_labels=True,
        width=list(weights)
    )
    f.savefig(plot_outpath)
    nx.write_weighted_edgelist(graph, edgelist_outpath)


def load_networkx_from_edgelist_file(fpath: Path) -> nx.Graph:
    tmp_g = nx.read_edgelist(fpath, nodetype=int, data=(("weight", float),))
    sai_graph = nx.Graph()
    sai_graph.add_nodes_from(sorted(tmp_g.nodes(data=True)))
    sai_graph.add_edges_from(tmp_g.edges(data=True))
    if sai_graph.get_edge_data(0, 1) == {}:
        for e in sai_graph.edges():
            sai_graph[e[0]][e[1]]['weight'] = 1
    return sai_graph


def _dynamic_networkx_allocator(
    nx_class: str,
    **kwargs: str,
) -> nx.Graph:
    sai_graph = getattr(nx, nx_class)(**kwargs)

    # TODO: there could be different ways to assign weights to edges. For the moment, we set all weights to 1
    # adding weights to edges
    for e in sai_graph.edges():
        sai_graph[e[0]][e[1]]['weight'] = 1

    return sai_graph


def _load_preset(preset: str) -> nx.Graph:
    sai_graph = getattr(nx, preset)()
    logging.info(f"NXGraph: loaded preset: {preset}")

    # forcing weights to edges
    for e in sai_graph.edges():
        # self.sai_graph[e[0]][e[1]]['weight'] = ra.uniform(0.01, 3)
        sai_graph[e[0]][e[1]]['weight'] = 1
    return sai_graph


@register_graph()
class NXGraph(Graph):
    def __init__(
        self,
        *args,
        experiment_cfg: ExperimentConfiguration,
        graph: Optional[nx.Graph] = None,
        fpath: Optional[Path] = None,
        preset: Optional[str] = None,
        nx_class: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(experiment_cfg)

        if graph is not None:
            self.nx_graph = graph
        elif fpath:
            if not fpath.exists():
                raise ValueError(f'Not found: {fpath}')
            logging.info(f"NXGraph, loading from file: {fpath}")
            self.nx_graph = load_networkx_from_edgelist_file(fpath)
        elif nx_class:
            logging.info(f"NXGraph, generated synthetic graph of type {nx_class} with parameters {kwargs}")
            self.nx_graph = _dynamic_networkx_allocator(nx_class, **kwargs)
        elif preset:
            logging.info(f"NXGraph, loaded preset: {preset}")
            self.nx_graph = _load_preset(preset)
        else:
            raise ValueError("Either specify fpath, preset or synth args")

    def dump(self, postfix: str = ""):
        outpath = self._experiment_cfg.graph_output_directory
        _nx_dump(
            self.nx_graph,
            plot_outpath=(outpath / f"graph{postfix}.pdf"),
            edgelist_outpath=(outpath / f"graph{postfix}.edgelist"),
        )

    @property
    def nodes(self):
        return [str(n) for n in self.nx_graph.nodes]

    def get_weight(self, source, neighbor):
        source = int(source)
        neighbor = int(neighbor)
        return self.nx_graph[source][neighbor]['weight']

    def neighbors(self, source):
        source = int(source)
        return [str(n) for n in self.nx_graph.neighbors(source)]
