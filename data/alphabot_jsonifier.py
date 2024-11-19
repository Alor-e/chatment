import json
import re

import yaml


def jsonify(dataset: str):
    intent_examples = {}
    header = []
    mapping = {}
    # Load the nlu.yml file, which contains the NLU data
    with open(f'data/datasets/{dataset}/{dataset}.yml', 'r') as f:
        nlu_data = yaml.load(f.read(), Loader=yaml.FullLoader)

        # For each intent, get the examples and their corresponding intents
        # and store them in a dictionary with the intent as the key
        for intent in nlu_data['nlu']:
            # If intent is not in the dictionary, add it to the header
            if "intent" not in intent:
                header.append(intent)
                continue
            intent_name = intent['intent']
            intent_examples[intent_name] = []
            # Every example has a \n at the end, so we split on that
            examples = intent['examples'].split("\n")
            for example in examples:
                original_example = example
                # Remove the "- " from the beginning of each example
                example = example[2:]
                if example == "":
                    continue
                # Remove square brackets
                example = example.replace("[", "").replace("]", "")
                # Remove the parentheses and the string inside them
                example = re.sub(r'\(.*?\)', '', example)
                # Add it to the list of examples
                intent_examples[intent_name].append(example)
                # Add the original example to the mapping
                mapping[example] = original_example

    with open('data/data.json', 'w') as f:
        json.dump(intent_examples, f, indent=4, sort_keys=True)
        print("JSON file created")

    return header, mapping
