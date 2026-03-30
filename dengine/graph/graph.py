from datetime import datetime
from typing import Optional, Sequence, Dict, Literal
from abc import abstractmethod, ABC

from dengine.config import ExperimentConfiguration


class Graph(ABC):
    def __init__(self, experiment_cfg: ExperimentConfiguration):
        self._experiment_cfg = experiment_cfg

    @abstractmethod
    def dump(self):
        ...

    @property
    @abstractmethod
    def nodes(self) -> Sequence:
        ...

    @abstractmethod
    def get_weight(self, source, neighbor) -> float:
        ...

    @abstractmethod
    def neighbors(self, source) -> Sequence:
        ...


class DynamicGraph(Graph, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def neighbors(self, source, time: Optional[datetime] = None) -> Sequence:
        ...

    @abstractmethod
    def contact_time(self, source, destination, time: datetime) -> float:
        ...

    def bulk_contact_times_end(
        self,
        links: Sequence[
            Dict[
                Literal["source", "destination", "UUID"],
                str
            ]
        ],
        time_start: datetime,
        time_end: datetime
    ) -> Dict[str, datetime]:
        res = {}
        for li in links:
            res[li["UUID"]] = self.contact_time_end(
                li["source"],
                li["destination"],
                time_start,
                time_end
            )
        return res

    def contact_time_end(self, source, destination, time_start: datetime, time_end: datetime | None = None) -> datetime:
        contact_time = self.contact_time(source, destination, time_start)
        contact_time_end = time_start.timestamp() + contact_time
        if time_end and time_end.timestamp() < contact_time_end:
            return time_end
        return datetime.fromtimestamp(contact_time_end)
