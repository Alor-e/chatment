import atexit
import os
import shutil
import subprocess
import time

import requests

import train_test as tt
from data import rasa_nlu_generator as nlu_gen


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RasaEntity(metaclass=Singleton):
    """
    Singleton for training, creating and interacting with a Rasa entity detector.
    """

    def __init__(self, training_x: list, training_y: list, header: str, mapping: dict, intents: dict):
        """
        Initializes the Rasa entity detector.
        """
        self.mapping = mapping
        self.yaml_generator = nlu_gen.RasaNLUGenerator(mapping, header)
        self.yaml_generator.generate_training(training_x, training_y, intents,
                                              output_path="./data/output/entity_training.yml")
        self.url = "http://localhost:5005/model/parse"
        self.headers = {"Content-Type": "text/plain"}
        atexit.register(self.cleanup)
        self.__train()
        self.__run()

    def cleanup(self) -> None:
        """
        Cleans up the Rasa entity detector.
        """
        self.process.kill()

    @staticmethod
    def __train() -> None:
        """
        Trains a rasa nlu for it.
        """
        try:
            os.mkdir("rasa-entity")
        except FileExistsError:
            pass
        shutil.copyfile("./data/domain.yml", "rasa-entity/domain.yml")
        shutil.copyfile("./data/output/entity_training.yml", "rasa-entity/nlu.yml")
        shutil.copyfile("./data/config.yml", "rasa-entity/config.yml")
        tt.train(domain_file_path="rasa-entity/domain.yml",
                 config_file_path="rasa-entity/config.yml",
                 training_file_path="rasa-entity/nlu.yml",
                 output_dir="rasa-entity")

    def __run(self) -> None:
        """
        Runs the NLU as a server to then be able to receive requests.
        """
        self.process = subprocess.Popen([shutil.which("rasa"), "run", "--enable-api", "--model", "rasa-entity"])
        self.__wait_for_server()

    def get_entities(self, text: str) -> list:
        """
        Gets entities from Rasa's NLU Server.

        :param text: Text to get entities from
        :return: List of entities present in the text
        """
        data = {"text": text}
        entities = requests.post(self.url, json=data, headers=self.headers).json()["entities"]
        unique_entities = set()
        for entity in entities:
            unique_entities.add(entity["entity"])
        return list(unique_entities)

    def __wait_for_server(self):
        """
        Waits for the server to be open and running.
        """
        print("Waiting for server to be open and running...")
        while True:
            try:
                requests.get(self.url)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        print("Server is open and running!")
