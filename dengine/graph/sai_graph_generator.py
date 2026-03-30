from warnings import deprecated

from .decorators import register_graph
from .nx_graph import NXGraph


@deprecated("This class has been refactored to NXGraph, please consider refactoring.")
@register_graph()
class SAIGraph(NXGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sai_graph = self.nx_graph
