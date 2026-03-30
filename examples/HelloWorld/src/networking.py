from datetime import datetime

import networkx as nx

from dengine import DynamicGraph
from dengine.graph.decorators import register_graph


@register_graph()
class CustomNetworking(DynamicGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nx_graph = nx.barabasi_albert_graph(50, 1, 42)

    def get_weight(self, source, neighbor):
        return self.nx_graph[source][neighbor]['weight']

    def dump(self):
        pass

    @property
    def nodes(self):
        return [str(x) for x in self.nx_graph.nodes]

    def neighbors(self, source, time=None):
        return [str(n) for n in self.nx_graph.neighbors(int(source))]

    def contact_time(self, source, destination, time: datetime) -> float:
        return 1
