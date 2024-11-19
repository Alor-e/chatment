from abc import ABC, abstractmethod
from snorkel.labeling.lf.core import LabelingFunction


class LabellingFunction(LabelingFunction, ABC):

    @abstractmethod
    def apply(self, data) -> int:
        pass
