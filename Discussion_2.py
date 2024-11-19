# Standard library imports
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import List

# Related third-party imports
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

# Snorkel imports
from snorkel.labeling.apply.core import LFApplier as SnorkelLFApplier
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

# Local application/library specific imports
import data.alphabot_jsonifier as data_jsonifier
import data.average_scores as av_sc
import data.loader as data_loader
import data.rasa_nlu_generator as rasa_nlu
import program.grouper as sem_grouper
import program.majority_voter as majority_voter
import program.run_config as rc
from data.average_scores import baseline, applied
from program.generator import LFGenerator
from program.pruner import Pruner
import train_test as tt


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
    output_dir = f"./logs/{dataset}/run_{run_idx}"
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


# Usage:
# Assume rc, loader, new_pos, new_labels, testing, testing_y, pruner_training_x, pruner_training_y, lfs, preds, majority_voter are defined elsewhere in your code.

# Generate the json data from the base YAML dataset, outputs data.json file
header, mapping = data_jsonifier.jsonify(rc.DATASET)

# Use seed or not for consistency in data splitting
if not rc.USE_SEED:
    random.seed(time.time_ns())
elif rc.USE_SEED:
    random.seed(rc.SEED)

for i in range(rc.NUMBER_OF_RUNS):

    if i == 0:
        # Generate random seed (using same seed as above)
        loader_seed = random.randint(0, 1000000)

        # Create Loader Class, loads the data.json file generated from the jsonifier.
        loader = data_loader.Loader()

        # Use loader class to split data
        loader.split_data_multi(split_ratio=rc.TRAINING_DATA_PERCENT, seed=i, testing_data_ratio=rc.TESTING_DATA_PERCENT)

        # Generate YAML data from json data splits (yaml data would now be split)
        generator = rasa_nlu.RasaNLUGenerator(mapping, header)

        # Generate Baseline training YAML data
        generator.generate_training(loader.training_x, loader.training_y, loader.intent_dict,
                                    output_path="./data/output/baseline_nlu.yml")
        
        # Generate Validation YAML data same for baseline and approach
        generator.generate_testing(loader.validation, loader.validation_y, loader.intent_dict,
                                output_path="./data/output/baseline_testing.yml")
        
        # Log out intent dict
        log_json(loader.intent_dict, rc.DATASET, i, "intent_dict.json")

        # Group together semantically similar intents
        grouper = sem_grouper.SemanticGrouper(loader.training_x, loader.training_y, loader.testing,
                                              loader.intent_dict,
                                              transformer=rc.TRANSFORMER)
        new_pos, new_labels = grouper.get_similar_sentences(threshold=rc.SEM_GROUP_THRESHOLD)

        log_json({"new_pos": new_pos, "new_labels": new_labels}, rc.DATASET, i, "grouper_output.json")

        # Expansion of small label labelled dataset
        training_x = loader.training_x + new_pos
        training_y = loader.training_y + new_labels

        amount_pruner_data = math.floor(len(training_x) * rc.PRUNER_DATA_PERCENT)
        if amount_pruner_data == 0:
            amount_pruner_data = 1

        training_x_reserved = training_x
        training_y_reserved = training_y

        # Data for Pruner Evaluation by selecting number of pruner percentage
        pruner_training_x = training_x[:amount_pruner_data]
        pruner_training_y = training_y[:amount_pruner_data]

        training_x = training_x[amount_pruner_data:]
        training_y = training_y[amount_pruner_data:]

        # Get index of testing data not in new_pos
        testing_index = [i for i in range(len(loader.testing)) if loader.testing[i] not in new_pos] 

        testing = [loader.testing[i] for i in testing_index]
        testing_y = [loader.testing_y[i] for i in testing_index]

        log_json({'intent_mapping': loader.intent_dict, 'training_x': loader.training_x, 'training_y': loader.training_y, 'testing': testing, 'testing_y': testing_y, 'validation': loader.validation, 'validation_y': loader.validation_y, 'pruner_training_x': pruner_training_x, 'pruner_training_y': pruner_training_y}, rc.DATASET, i, "loader_dict.json")

        # new fast test
        testing = loader.testing
        testing_y = loader.testing_y

        testing.extend(loader.lf_testing)
        testing_y.extend(loader.lf_testing_y)

        # Use the function
        LFGenerator.generate_domain_file("./data/output/baseline_nlu.yml", "./rasa/entity_domain.yml")

        tt.train(domain_file_path="./rasa/entity_domain.yml",
                config_file_path="./rasa/config.yml",
                training_file_path="./data/output/baseline_nlu.yml",
                output_dir=f"./entity_nlu/{rc.DATASET}/run_{i}")

        # Generate the labelling functions
        lf_generator = LFGenerator(training_x, training_y, testing, testing_y, loader.all_intents, f"./entity_nlu/{rc.DATASET}/run_{i}")
        lfs = lf_generator.generate_lfs(rc.MAX_WORDS_PER_LF)

        log_json({"number of labelling functions": len(lfs), "lf_words_labels": [{"words": item.words, "labels": item.label} if hasattr(item, 'words') else {"entity": item.entity, "labels": item.label} if hasattr(item, 'entity') else {"word_entity_combo": item.word_entity_combo, "labels": item.label} if hasattr(item, 'word_entity_combo') else {"ml": None, "labels": item.label} for item in lfs]}, rc.DATASET, i, "lfs_output.json")

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

        pruned_lfs_index = pruner.prune_lfs_threshold_n_classes(main_iteration = i)
        pruned_lfs = [lfs[index] for index in pruned_lfs_index]

        if not pruned_lfs:
            print("pruned_lfs is empty")
            continue

        print('PRUNED', pruned_lfs)

    else:
        random.shuffle(pruned_lfs)

    for count in range(len(pruned_lfs)):
        sub_lfs = pruned_lfs[:count+1]

        name_count = "D2_" + str(count)
        rc.OUTPUT_PATH = './results/' + rc.DATASET + '_' + name_count + '/'

        sApplier = SnorkelLFApplier(sub_lfs)
        lf_matrix = sApplier.apply(pruner_training_x)

        # Generate LFAnalysis

        pruner_training_y = np.array(pruner_training_y, dtype=int)
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
        lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in sub_lfs]

        # Add number_of_queries as a new column in lf_analysis
        lf_summary_df['number_of_queries'] = lf_number_of_queries

        # Add class labels
        lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in sub_lfs]
        lf_summary_df['class_labels'] = lf_class_labels

        lf_summary_json = lf_summary_df.to_json(orient='records')

        try:
            lf_summary_df.to_csv(f'./logs/{rc.DATASET}_{name_count}/run_{i}/lf_summary.csv', index=False)
        except OSError:
            os.makedirs(f'./logs/{rc.DATASET}_{name_count}/run_{i}', exist_ok = True)
            lf_summary_df.to_csv(f'./logs/{rc.DATASET}_{name_count}/run_{i}/lf_summary.csv', index=False)

        # Log out intent dict
        output_dir = f"./logs/{rc.DATASET}_{name_count}/run_{i}"
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
        # Print percent of correct predictions
        print("Percentage of correct predictions: ", majority_voter.get_accuracy(testing_y, preds))

        log_json({"preds": preds.tolist()}, f"{rc.DATASET}_{name_count}", i, "mlv_output_t.json")
        log_json({"accuracy": majority_voter.get_accuracy(testing_y, preds)}, f"{rc.DATASET}_{name_count}", i, "mlv_accuracy_output_t.json")

        # DATA LABELLING ON TEST DATASET AND VALIDATION DATASET
        lf_matrix = sApplier.apply(testing + loader.validation)

        MLV = MajorityLabelVoter(len(loader.all_intents))
        preds = MLV.predict(lf_matrix)
        # Print percent of correct predictions
        print("Percentage of correct predictions: ", majority_voter.get_accuracy(testing_y + loader.validation_y, preds))

        # Log MLV output and accuracy
        log_json({"preds": preds.tolist()}, f"{rc.DATASET}_{name_count}", i, "mlv_output.json")
        log_json({"accuracy": majority_voter.get_accuracy(testing_y + loader.validation_y, preds)}, f"{rc.DATASET}_{name_count}", i, "mlv_accuracy_output.json")

        auc_score = calculate_average_auc(testing_y + loader.validation_y, preds)
        log_json({"auc_score": auc_score}, f"{rc.DATASET}_{name_count}", i, "mlv_auc_score.json")


import os
import json
import glob
import numpy as np

def get_average_auc_score(prune_dir_pattern='./logs/*D2*/'):
    average_scores = {}

    # Step 2: Find all directories with 'prune' in their names within the logs directory.
    prune_directories = glob.glob(prune_dir_pattern)

    # Step 3: For each directory, find all 'mlv_auc_score.json' files for each run.
    for directory in prune_directories:
        auc_scores = []
        json_files = glob.glob(os.path.join(directory, 'run_*', 'mlv_auc_score.json'))

        # Step 4: Read each JSON file and extract the auc score.
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                # Assuming that the JSON structure is { "mlv_auc_score": value }
                auc_score = data.get("auc_score")
                if auc_score is not None:
                    auc_scores.append(auc_score)

        # Step 5: Calculate the average score if there are any scores.
        if auc_scores:
            average_score = np.mean(auc_scores)
            average_scores[directory[len('./logs/'):-1]] = average_score

    # Step 7: Print the dictionary of average scores.
    print(json.dumps(average_scores, indent=4))

# Execute the function
get_average_auc_score()
