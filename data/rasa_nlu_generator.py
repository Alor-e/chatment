import yaml


def should_use_block(value):
    for c in u"\u000a\u000d\u001c\u001d\u001e\u0085\u2028\u2029":
        if c in value:
            return True
    return False


def my_represent_scalar(self, tag, value, style=None):
    if style is None:
        if should_use_block(value):
            style = '|'
        else:
            style = self.default_style

    node = yaml.representer.ScalarNode(tag, value, style=style)
    if self.alias_key is not None:
        self.represented_objects[self.alias_key] = node
    return node


class RasaNLUGenerator:
    """
    Class to generate the Rasa NLU yaml file from the passed in data.
    Takes in a list of tuples, with the first element being the example and the second element being the label.
    """

    def __init__(self, mapping, header: str = None):
        self.header = header
        self.mapping = mapping

    def generate_training(self, examples, labels, intents: dict, output_path: str = './data/output/nlu.yml') -> None:
        """
        Generates the yaml nlu file from the data.

        :param examples: List of examples
        :param labels: List of labels
        :param intent: Intent name
        :param output_path: The path to the output file.
        """
        self.__generate_yaml(examples, intents, labels, output_path)

    def generate_testing(self, examples, labels, intents: dict,
                         output_path: str = './data/output/test_data.yml') -> None:
        """
        Generates the yaml nlu file from the data.

        :param examples:
        :param labels:
        :param intent:
        :param output_path:
        :return:
        """
        self.__generate_yaml(examples, intents, labels, output_path)

    def __generate_yaml(self, examples, intents: dict, labels, output_path) -> None:
        """
        Generates the yaml file as a generic pass in.

        :param examples: List of examples
        :param intent: Intent name
        :param labels: List of labels
        :param output_path: The path to the output file.
        """
        document = {"nlu": []}
        if self.header is not None:
            for header in self.header:
                document["nlu"].append(header)

        for intent, intent_id in intents.items():
            pos_examples = ""
            for example, label in zip(examples, labels):
                if label == intent_id:
                    pos_examples += f"{self.mapping[example]}\n"
            document["nlu"].append({"intent": intent, "examples": pos_examples})

        with open(output_path, 'w') as f:
            yaml.representer.BaseRepresenter.represent_scalar = my_represent_scalar
            yaml.dump(document, f, default_flow_style=False, sort_keys=False)
            print("YAML file created")
