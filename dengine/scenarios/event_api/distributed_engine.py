from dengine.scenarios.decorators import register_scenario

from .sync_engine import SyncEngine


@register_scenario()
class DistributedEngine(SyncEngine):
    pass
