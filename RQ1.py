import json
import math
import os
import pickle
import random
import shutil
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from snorkel.labeling.apply.core import LFApplier as SnorkelLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.analysis import LFAnalysis

import data.alphabot_jsonifier as data_jsonifier
import data.average_scores as av_sc
import data.loader as data_loader
import data.rasa_nlu_generator as rasa_nlu
import program.grouper as sem_grouper
import program.majority_voter as majority_voter
from program.pruner import Pruner
import program.run_config as rc
import train_test as tt
from data.average_scores import baseline, applied
from program.generator import LFGenerator
from snorkel.labeling.model import LabelModel
from typing import List
from sentence_transformers import SentenceTransformer



def generate_random_numbers(list_length: int, number_of_classes: int):
    """
    Parameters:
    - list_length: Number of random numbers
    - number_of_classes: Polarity of random numbers.
    """
    random_numbers = []  # List to store the generated random numbers
    for _ in range(list_length):  # Loop x times
        num = random.randint(0, number_of_classes)  # Generate a random number between 0 and n
        random_numbers.append(num)  # Add the random number to the list
    return random_numbers


def serialize_lfs_and_models(lfs, file_path):
    """
    Serializes labeling functions and any associated models to a binary file.

    Parameters:
    - lfs: The list of labeling functions and models to serialize.
    - file_path: Path to the file where the objects will be stored.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(lfs, file)


def deserialize_lfs_and_models(file_path):
    """
    Deserializes labeling functions and models from a binary file.

    Parameters:
    - file_path: Path to the file from which the objects will be loaded.
    
    Returns:
    - The deserialized list of labeling functions and models.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def log_json(data, dataset, run_idx, file_name, sub_dir=None):
    """
    Logs data to a JSON file.

    :param data: The data to be logged.
    :param dataset: The dataset being used.
    :param run_idx: The run index.
    :param file_name: The name of the file to write.
    :param sub_dir: An optional sub-directory under ./logs/{dataset}/run_{run_idx}.
    """
    # Construct the output directory path
    output_dir = f"./logs/{dataset}_random/run_{run_idx}"
    if sub_dir:
        output_dir = os.path.join(output_dir, sub_dir)
    
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Construct the output file path
    output_file = os.path.join(output_dir, file_name)
    
    # Write data to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def calculate_average_auc(gold_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculate the average AUC for a multi-class classification problem.
    
    Parameters:
    - gold_labels (List[int]): A list of the actual labels.
    - predicted_labels (List[int]): A list of the predicted labels.
    
    Returns:
    - float: The average AUC.
    """
    # Initialize a variable to store the sum of AUCs
    sum_auc = 0.0
    
    # Unique labels in the dataset
    unique_labels = np.unique(gold_labels)
    
    # Calculate AUC for each class against all other classes
    for label in unique_labels:
        # Convert the multi-class labels into binary labels: 'label' vs 'all others'
        binary_gold_labels = [1 if x == label else 0 for x in gold_labels]
        binary_predicted = [1 if x == label else 0 for x in predicted_labels]
        
        # Compute the AUC for this binary classification task
        auc = roc_auc_score(binary_gold_labels, binary_predicted)
        sum_auc += auc
    
    # Calculate the average AUC
    average_auc = sum_auc / len(unique_labels)

    return average_auc


# Jsonify the data, and get the mappings to the original data and the header of the file
header, mapping = data_jsonifier.jsonify(rc.DATASET)

model = SentenceTransformer(rc.MODEL_LOCATION)

for i in range(rc.NUMBER_OF_RUNS):
    # Load the data from JSON file.
    loader_seed = random.randint(0, 1000000)
    loader = data_loader.Loader()

    loader.split_data_multi(split_ratio=rc.TRAINING_DATA_PERCENT, seed=i, testing_data_ratio=rc.TESTING_DATA_PERCENT)

    generator = rasa_nlu.RasaNLUGenerator(mapping, header)
    generator.generate_training(loader.training_x, loader.training_y, loader.intent_dict,
                                output_path="./data/output/baseline_nlu.yml")
    generator.generate_testing(loader.validation, loader.validation_y, loader.intent_dict,
                               output_path="./data/output/baseline_testing.yml")
    
    # Log out intent dict
    output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "intent_dict.json")
    with open(output_file, "w") as f:
        json.dump(loader.intent_dict, f, indent=4)

    # # Group together semantically similar intents
    grouper = sem_grouper.SemanticGrouper(loader.training_x, loader.training_y, loader.testing,
                                          intent_dict=loader.intent_dict,
                                          transformer=rc.TRANSFORMER
                                          )
    new_pos, new_labels = grouper.get_similar_sentences(threshold=rc.SEM_GROUP_THRESHOLD)

    # Log grouper output
    output_dir = f"./logs/{rc.DATASET}/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "grouper_output.json")
    with open(output_file, "w") as f:
        json.dump({"new_pos": new_pos, "new_labels": new_labels}, f, indent=4)

    # Expansion of small label labelled dataset
    training_x = loader.training_x + new_pos
    training_y = loader.training_y + new_labels

    amount_pruner_data = math.floor(len(training_x) * rc.PRUNER_DATA_PERCENT)
    if amount_pruner_data == 0:
        amount_pruner_data = 1

    training_x_reserved = training_x
    training_y_reserved = training_y

    pruner_training_x = training_x[:amount_pruner_data]
    pruner_training_y = training_y[:amount_pruner_data]

    training_x = training_x[amount_pruner_data:]
    training_y = training_y[amount_pruner_data:]

    # Get index of testing data not in new_pos
    testing_index = [i for i in range(len(loader.testing)) if loader.testing[i] not in new_pos]

    testing = [loader.testing[i] for i in testing_index]
    testing_y = [loader.testing_y[i] for i in testing_index]

    # Log out loader dict
    output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "loader_dict.json")
    with open(output_file, "w") as f:
        json.dump({'intent_mapping': loader.intent_dict, 'training_x': loader.training_x, 'training_y': loader.training_y, 'testing': testing, 'testing_y': testing_y, 'validation': loader.validation, 'validation_y': loader.validation_y, 'pruner_training_x': pruner_training_x, 'pruner_training_y': pruner_training_y}, f, indent=4)

    # Use the function
    LFGenerator.generate_domain_file("./data/output/baseline_nlu.yml", "./rasa/entity_domain.yml")

    tt.train(domain_file_path="./rasa/entity_domain.yml",
            config_file_path="./rasa/config.yml",
            training_file_path="./data/output/baseline_nlu.yml",
            output_dir=f"./entity_nlu/{rc.DATASET}/run_{i}")

    # Generate the labelling functions
    lf_generator = LFGenerator(training_x, training_y, testing, testing_y, loader.all_intents, f"./entity_nlu/{rc.DATASET}/run_{i}")
    lfs = lf_generator.generate_lfs(rc.MAX_WORDS_PER_LF)
    serialize_lfs_and_models(lfs, f"./logs/{rc.DATASET}/run_{i}/lfs.pkl")
    print("Number of labelling functions: ", len(lfs))

    # Log labelling functions
    output_dir = f"./logs/{rc.DATASET}/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "lfs_output.json")
    
    lf_words_labels = [{"words": item.words, "labels": item.label} if hasattr(item, 'words') else {"entity": item.entity, "labels": item.label} if hasattr(item, 'entity') else {"word_entity_combo": item.word_entity_combo, "labels": item.label} if hasattr(item, 'word_entity_combo') else {"ml": None, "labels": item.label} for item in lfs]
    lf_log = {"number of labelling functions": len(lfs), "lf_words_labels": lf_words_labels}
    with open(output_file, "w") as f:
        json.dump(lf_log, f, indent=4)

    sApplier = SnorkelLFApplier(lfs)
    lf_matrix = sApplier.apply(pruner_training_x)
    lf_summary_df = LFAnalysis(lf_matrix).lf_summary(np.array(pruner_training_y, dtype=int))

    # fetch the number of queries per lf
    lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in lfs]

    # Add number_of_queries as a new column in lf_analysis
    lf_summary_df['number_of_queries'] = lf_number_of_queries

    # Add class labels
    lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in lfs]
    lf_summary_df['class_labels'] = lf_class_labels

    pruner = Pruner(lfs=lfs, pruner_training_x=pruner_training_x, pruner_training_y=pruner_training_y, number_of_classes=len(loader.intent_dict.values()))

    # Filter your list of LFs
    pruned_lfs_index = pruner.prune_lfs_threshold_n_classes(max_lf=len(loader.intent_dict.values())*5, min_lf=len(loader.intent_dict.values()), threshold=rc.PRUNER_SCORE_THRESHOLD, main_iteration=i)

    pruned_lfs = [lfs[index] for index in pruned_lfs_index]

    def rq_logs():

        sApplier = SnorkelLFApplier(pruned_lfs)
        lf_matrix_prediction_logs = sApplier.apply(testing + loader.validation)

        gold_standard_labels = testing_y + loader.validation_y

        rq_main_log = {
            "intent_label_mapping": loader.intent_dict, 
            "gold_standard_labels": gold_standard_labels,
            "logs": []
        }

        for count, (lf, index) in enumerate(zip(pruned_lfs, pruned_lfs_index)):
            lf_index = index
            lf_type = type(lf).__name__
            lf_pattern_basis = lf.pattern_basis

            # Get the labels from the labeling function
            lf_predicted_labels = list(lf_matrix_prediction_logs[:, count])
            lf_predicted_labels = [int(x) for x in lf_predicted_labels]
            
            # Get the indices where the LF didn't abstain
            non_abstain_indices = np.where(np.array(lf_predicted_labels) != -1)[0]
            non_abstain_lf_labels = [lf_predicted_labels[i] for i in non_abstain_indices]
            non_abstain_gold_labels = [gold_standard_labels[i] for i in non_abstain_indices]

            # Compute the F1 score for the non-abstained labels
            f1_main = f1_score(non_abstain_gold_labels, non_abstain_lf_labels, average='weighted')

            # Find all unique classes in the gold labels
            classes = list(set(gold_standard_labels))

            # Initialize the dictionary
            class_dict = defaultdict(lambda: defaultdict(list))

            for c in classes:
                # Get the indices where the gold label is the current class
                indices = [idx for idx, label in enumerate(gold_standard_labels) if label == c]
                
                # Get the corresponding gold labels and LF labels
                class_gold_labels = [gold_standard_labels[idx] for idx in indices]
                class_lf_labels = [lf_predicted_labels[idx] for idx in indices]
                
                # Only consider data points where the LF didn't abstain
                non_abstain_indices = [idx for idx, label in enumerate(class_lf_labels) if label != -1]
                non_abstain_class_gold_labels = [class_gold_labels[idx] for idx in non_abstain_indices]
                non_abstain_class_lf_labels = [class_lf_labels[idx] for idx in non_abstain_indices]

                # Convert numpy.int64 to int
                non_abstain_class_gold_labels = [int(label) for label in non_abstain_class_gold_labels]
                non_abstain_class_lf_labels = [int(label) for label in non_abstain_class_lf_labels]
                
                # Compute the F1 score for the non-abstained labels
                if non_abstain_class_gold_labels and non_abstain_class_lf_labels:  # Check if lists are not empty
                    f1 = f1_score(non_abstain_class_gold_labels, non_abstain_class_lf_labels, average='weighted', pos_label=None)
                else:
                    f1 = None
                
                # Store the gold labels, LF labels, and F1 score for the current class
                class_dict[c]['gold'] = non_abstain_class_gold_labels
                class_dict[c]['predicted'] = non_abstain_class_lf_labels
                class_dict[c]['f1'] = f1

            number_of_queries = lf.number_of_queries

            log_data = {
                "index_of_lf": int(lf_index),
                "type_of_lf": str(lf_type),
                "pattern_basis": lf_pattern_basis,
                "f1": f1_main,
                "predicted_labels": lf_predicted_labels,
                "per_class_values": class_dict,
                "number_of_queries": int(number_of_queries),
                "polarity": lf.label
            }
            # print(log_data)
            rq_main_log["logs"].append(log_data)

        # Log out intent dict
        output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, "rq_logs.json")

        with open(output_file, "w") as f:
            json.dump(rq_main_log, f, indent=4)

    # generate rq logs
    rq_logs()

    if not pruned_lfs:
        print("pruned_lfs is empty")
        print("55555555555555555555555555555555555555555555555555545555")
    else:
        sApplier = SnorkelLFApplier(pruned_lfs)
        lf_matrix = sApplier.apply(pruner_training_x)

    # sApplier = SnorkelLFApplier(pruned_lfs)
    print('PRUNED', pruned_lfs)

    # Generate LFAnalysis

    pruner_training_y = np.array(pruner_training_y, dtype=int) # wrong use validation not testing
    label_conflict = LFAnalysis(lf_matrix).label_conflict()
    label_coverage = LFAnalysis(lf_matrix).label_coverage()
    label_overlap = LFAnalysis(lf_matrix).label_overlap()

    lf_conflicts = LFAnalysis(lf_matrix).lf_conflicts()
    lf_conflicts_normalized = LFAnalysis(lf_matrix).lf_conflicts(normalize_by_overlaps=True)

    lf_coverages = LFAnalysis(lf_matrix).lf_coverages()

    lf_empirical_accuracies = LFAnalysis(lf_matrix).lf_empirical_accuracies(pruner_training_y)
    lf_empirical_probs = LFAnalysis(lf_matrix).lf_empirical_probs(pruner_training_y, len(loader.intent_dict.values()))

    lf_overlaps = LFAnalysis(lf_matrix).lf_overlaps()
    lf_overlaps_normalized = LFAnalysis(lf_matrix).lf_overlaps(normalize_by_coverage=True)

    lf_polarities = LFAnalysis(lf_matrix).lf_polarities()
    lf_summary_df = LFAnalysis(lf_matrix).lf_summary(pruner_training_y)

    # fetch the number of queries per lf
    lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in pruned_lfs]

    # Add number_of_queries as a new column in lf_analysis
    lf_summary_df['number_of_queries'] = lf_number_of_queries

    # Add class labels
    lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in pruned_lfs]
    lf_summary_df['class_labels'] = lf_class_labels

    lf_summary_json = lf_summary_df.to_json(orient='records')
    try:
        lf_summary_df.to_csv(f'./logs/{rc.DATASET}_random/run_{i}/lf_summary.csv', index=False)
    except OSError:
        os.makedirs(f'./logs/{rc.DATASET}_random/run_{i}', exist_ok = True)
        lf_summary_df.to_csv(f'./logs/{rc.DATASET}_random/run_{i}/lf_summary.csv', index=False)

    # Log out intent dict
    output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "lf_analysis.json")

    with open(output_file, "w") as f:
        json.dump({
            "number of labelling functions": len(lfs),
            "number_of_labels": len(loader.intent_dict.values()),
            "intent_label_mapping": loader.intent_dict, 
            "label_conflict": label_conflict,
            "label_coverage": label_coverage,
            "label_overlap": label_overlap,
            "lf_conflicts": lf_conflicts.tolist(),
            "lf_conflicts_normalized": lf_conflicts_normalized.tolist(),
            "lf_coverages": lf_coverages.tolist(),
            "lf_empirical_accuracies": lf_empirical_accuracies.tolist(),
            "lf_empirical_probs": lf_empirical_probs.tolist(),
            "lf_overlaps": lf_overlaps.tolist(),
            "lf_overlaps_normalized": lf_overlaps_normalized.tolist(),
            "lf_polarities": [[int(x) for x in inner_list] for inner_list in lf_polarities]
        }, f, indent=4)

    with open(os.path.join(output_dir, 'lf_summary.json'), 'w') as f:
        json.dump(lf_summary_json, f)


    # DATA LABELLING ON TEST DATASET
    lf_matrix = sApplier.apply(testing)

    MLV = MajorityLabelVoter(len(loader.all_intents))
    preds = MLV.predict(lf_matrix)
    
    # Log MLV output and accuracy
    output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "mlv_output_t.json")
    with open(output_file, "w") as f:
        json.dump({"preds": preds.tolist()}, f, indent=4)

    # DATA LABELLING ON TEST DATASET AND VALIDATION DATASET
    lf_matrix = sApplier.apply(testing + loader.validation)

    MLV = MajorityLabelVoter(len(loader.all_intents))
    preds = MLV.predict(lf_matrix)
    # Print percent of correct predictions
    print("Percentage of correct predictions: ", majority_voter.get_accuracy(testing_y + loader.validation_y, preds))

    # Log MLV output and accuracy
    output_dir = f"./logs/{rc.DATASET}_random/run_{i}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, "mlv_output.json")
    with open(output_file, "w") as f:
        json.dump({"preds": preds.tolist()}, f, indent=4)

    accuracy_output_file = os.path.join(output_dir, "mlv_accuracy_output.json")
    with open(accuracy_output_file, "w") as f:
        json.dump({"accuracy": majority_voter.get_accuracy(testing_y + loader.validation_y, preds)}, f, indent=4)

    auc_score_value = calculate_average_auc(testing_y + loader.validation_y, preds)
    log_json({"auc_score": auc_score_value}, f"{rc.DATASET}", i, "mlv_auc_score.json")

    # Add F1 score calculation and logging
    f1_score_value = f1_score(testing_y + loader.validation_y, preds, average='weighted')
    log_json({"f1_score": f1_score_value}, f"{rc.DATASET}", i, "mlv_f1_score.json")

    # Add precision score calculation and logging
    precision_score_value = precision_score(testing_y + loader.validation_y, preds, average='weighted')
    log_json({"precision_score": precision_score_value}, f"{rc.DATASET}", i, "mlv_precision_score.json")

    # Add recall score calculation and logging
    recall_score_value = recall_score(testing_y + loader.validation_y, preds, average='weighted')
    log_json({"recall_score": recall_score_value}, f"{rc.DATASET}", i, "mlv_recall_score.json")
