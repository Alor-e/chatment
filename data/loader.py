import json
import math
import random
import itertools
from typing import Tuple


def two_list_shuffle(data_x, data_y, my_random) -> Tuple[list, list]:
    """
    Shuffles the data.

    :param data_x: First list to shuffle
    :param data_y: Second list to shuffle
    :param my_random: Random number generator
    :return:
    """
    temp = list(zip(data_x, data_y))
    my_random.shuffle(temp)
    data_x, data_y = zip(*temp)
    return list(data_x), list(data_y)


class Loader:
    """
    Loads the data as training_x, testing, and validation data.
    """

    def __init__(self, data_path: str = 'data/data.json'):
        """
        :param data_path: The data path to the json file.
        """
        self.training_x = None
        self.training_y = None

        self.testing = None
        # Used only for testing purposes
        self.testing_y = None

        # Testing also split into 2 parts
        self.grouper_testing = None
        # Used only for testing purposes
        self.grouper_testing_y = None
        self.lf_testing = None
        # Used only for testing purposes
        self.lf_testing_y = None

        self.validation = None
        # Used only for testing purposes
        self.validation_y = None

        self.intent = None
        self.intent_dict = {}
        self.__data_path = data_path
        self.__data = self.__load_data()
        self.all_intents = list(self.__data.keys())

    def __load_data(self) -> dict:
        """
        Loads the data from the json file as a dictionary.

        :return: Python dictionary with the data from the json file.
        """
        with open(self.__data_path, 'r') as f:
            data = json.load(f)
        return data

    def __split_intent(self, data: list, label: int, ratio: float, my_random: random.Random, testing_data_ratio: float = 0.7) -> Tuple[
        list, list, list, list, list, list]:
        """
        Splits a single intent into training, testing and validation

        :param data: The data of the intent
        :param label: The label of the intent
        :param ratio: Split ratio, 0.1 = 10%
        :return: training_x, training_y, testing_x, testing_y, validation_x, validation_y
        """

        training_x = []
        training_y = []
        testing_x = []
        testing_y = []
        validation_x = []
        validation_y = []

        amount_in_training = math.floor(len(data) * ratio)
        if amount_in_training == 0:
            amount_in_training = 1 

        # old method that split testing and validation data in two equal halves
        # amount_in_testing = (len(data) - amount_in_training) // 2
        # amount_in_validation = len(data) - amount_in_training - amount_in_testing

        amount_in_testing = math.floor(len(data) * testing_data_ratio)
        if amount_in_testing == 0:
            amount_in_testing = 1

        amount_in_validation = len(data) - amount_in_training - amount_in_testing # set whatever is left from testing to validation

        # Shuffle the data
        my_random.shuffle(data)

        # Split the data
        training_x.extend(data[:amount_in_training])
        training_y.extend([label] * amount_in_training)
        testing_x.extend(data[amount_in_training:amount_in_training + amount_in_testing])
        testing_y.extend([label] * amount_in_testing)
        validation_x.extend(data[amount_in_training + amount_in_testing:])
        validation_y.extend([label] * amount_in_validation)

        return training_x, training_y, testing_x, testing_y, validation_x, validation_y

    def split_data_multi(self, split_ratio: float = 0.1, seed: int = 1234, testing_data_ratio: float = 0.7) -> None:
        """
        Splits the data into training, testing, and validation data.

        :param split_ratio: The ratio to split for training data
        :param seed: The seed for the random number generator.
        """
        my_random = random.Random(seed)

        training_x = []
        training_y = []
        testing_x = []
        testing_y = []
        validation_x = []
        validation_y = []

        counter = 0
        for intent in self.all_intents:
            data_x = self.__data[intent]
            self.intent_dict[intent] = counter
            tr_x, tr_y, ts_x, ts_y, v_x, v_y = self.__split_intent(data_x, counter, split_ratio, my_random, testing_data_ratio)
            training_x.extend(tr_x)
            training_y.extend(tr_y)
            testing_x.extend(ts_x)
            testing_y.extend(ts_y)
            validation_x.extend(v_x)
            validation_y.extend(v_y)
            counter += 1

        self.training_x = training_x
        self.training_y = training_y

        # Split testing into grouper part and LF part
        self.testing = testing_x
        self.testing_y = testing_y

        self.grouper_testing = self.testing[:len(self.testing) // 2]
        self.grouper_testing_y = self.testing_y[:len(self.testing_y) // 2]
        self.lf_testing = self.testing[len(self.testing) // 2:]
        self.lf_testing_y = self.testing_y[len(self.testing_y) // 2:]

        self.validation = validation_x
        self.validation_y = validation_y

    def split_data_single(self, intent: str, split_ratio: float = 0.1, seed: int = 1234) -> None:
        """
        Splits the data into training, testing, and validation data.

        :param intent: The intent to split the data for.
        :param split_ratio: The ratio of the training_x data.
        :param seed: The seed for the random number generator.
        """
        self.intent = intent
        my_random = random.Random(seed)
        ratio = math.floor(split_ratio * 100)

        # Split the data into positive and negative data
        current_intent_data_x = self.__data[intent]
        current_intent_data_y = [1] * len(current_intent_data_x)
        rest_of_data = {i: self.__data[i] for i in self.__data if i != self.intent}
        rest_of_data_x = list(itertools.chain.from_iterable(list(rest_of_data.values())))
        rest_of_data_y = [0] * len(rest_of_data_x)

        # Shuffle the data
        my_random.shuffle(current_intent_data_x)
        my_random.shuffle(rest_of_data_x)

        # Split the data into training, testing, and validation data
        data_ratio = len(current_intent_data_x) // ratio

        # At least start with 1 data point
        if data_ratio == 0:
            data_ratio = 1

        rest_ratio = len(rest_of_data_x) // ratio
        self.training_x = \
            current_intent_data_x[:data_ratio] + rest_of_data_x[:rest_ratio]
        self.training_y = \
            current_intent_data_y[:data_ratio] + rest_of_data_y[:rest_ratio]

        rest_x = \
            current_intent_data_x[data_ratio:] + rest_of_data_x[rest_ratio:]
        rest_y = \
            current_intent_data_y[data_ratio:] + rest_of_data_y[rest_ratio:]

        temp = list(zip(current_intent_data_x[data_ratio:], current_intent_data_y[data_ratio:]))
        my_random.shuffle(temp)
        current_intent_data_x, current_intent_data_y = zip(*temp)
        current_intent_data_x = list(current_intent_data_x)
        current_intent_data_y = list(current_intent_data_y)

        temp = list(zip(rest_of_data_x[rest_ratio:], rest_of_data_y[rest_ratio:]))
        my_random.shuffle(temp)
        rest_of_data_x, rest_of_data_y = zip(*temp)
        rest_of_data_x = list(rest_of_data_x)
        rest_of_data_y = list(rest_of_data_y)

        testing_data_x = current_intent_data_x[:len(current_intent_data_x) // 2]
        testing_data_y = current_intent_data_y[:len(current_intent_data_y) // 2]
        validation_data_x = current_intent_data_x[len(current_intent_data_x) // 2:]
        validation_data_y = current_intent_data_y[len(current_intent_data_y) // 2:]
        testing_data_x += rest_of_data_x[:len(rest_of_data_x) // 2]
        testing_data_y += rest_of_data_y[:len(rest_of_data_y) // 2]
        validation_data_x += rest_of_data_x[len(rest_of_data_x) // 2:]
        validation_data_y += rest_of_data_y[len(rest_of_data_y) // 2:]

        self.testing = testing_data_x
        self.testing_y = testing_data_y

        self.validation = validation_data_x
        self.validation_y = validation_data_y
