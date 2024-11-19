from abc import ABC, abstractmethod
from program.labelling_functions.LabellingFunction import LabellingFunction


class MLLabeller(LabellingFunction, ABC):

    @abstractmethod
    def fit(self, data, labels) -> None:
        """
        Fits the model to the data

        :param data: Data to be used for training, in unigrams.
        :param labels: Labels for the data.
        """
        pass

    @abstractmethod
    def apply(self, data) -> int:
        """
        Applies the model to the data.

        :param data: Data to apply on, in unigrams.
        :return: Matrix of the data with the model applied, rows are the sentences, columns are the labels.
        """
        pass
