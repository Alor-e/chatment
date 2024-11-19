import yaml
import pandas as pd

def read_yaml_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

            if file_path.endswith("askgit/askgit.yml"):
                dataset = yaml_data['nlu'][2:]
            else:
                dataset = yaml_data['nlu'][:]
            return dataset
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    

def get_total_instances(dataset):
    ...

def total_examples_count(data):
    total_count = sum(len(intent['examples'].split('\n')) for intent in data)
    return total_count - 1

def examples_per_intent(data):
    intent_counts = {}
    for intent in data:
        intent_name = intent['intent']
        example_count = len(intent['examples'].split('\n'))
        intent_counts[intent_name] = example_count - 1
    return intent_counts


def data_label_ratio(data):
    total_examples = total_examples_count(data)
    label_counts = examples_per_intent(data)
    label_ratios = {}
    for intent, count in label_counts.items():
        ratio = count / total_examples
        label_ratios[intent] = ratio * 100
    return label_ratios

base_url = "data/datasets/"
file_path = base_url + "askgit/askgit.yml"
dataset = read_yaml_file(file_path)

datasets = [base_url + item for item in ["askgit/askgit.yml", "ubuntu/ubuntu.yml", "msa/msa.yml", "sof/sof.yml"]]

for item_path in datasets:
    dataset = read_yaml_file(item_path)
    class_ratio = data_label_ratio(dataset)
    number_per_intent = examples_per_intent(dataset)

    keys = list(class_ratio.keys())
    values = list(class_ratio.values())
    number = list(number_per_intent.values())

    df = pd.DataFrame({'Class': keys, 'Ratio (%)': values, 'Number of Queries': number}).sort_values("Ratio (%)", ascending=False).reset_index()
    print('\n\n', item_path.split("/")[-2])
    print(df)

