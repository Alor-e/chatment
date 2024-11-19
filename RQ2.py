# Standard library imports
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

# Related third-party imports
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

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
    # grouper = sem_grouper.SemanticGrouper(loader.training_x, loader.training_y, loader.testing, # changed loader.grouper_testing to loader.testing
    #                                       loader.intent_dict,
    #                                       transformer=rc.TRANSFORMER)
    # new_pos, new_labels = grouper.get_similar_sentences(threshold=rc.SEM_GROUP_THRESHOLD) # new data labels added grouper_testing serves as the basis

    # log_json({"new_pos": new_pos, "new_labels": new_labels}, rc.DATASET, i, "grouper_output.json")

    with open(f"./logs/{rc.DATASET}_mlv_lm/run_{i}/grouper_output.json", 'r') as json_file:
        json_data: dict = json.load(json_file)

    new_pos = json_data["new_pos"]
    new_labels = json_data["new_labels"]

    # Expansion of small label labelled dataset
    training_x = loader.training_x + new_pos
    training_y = loader.training_y + new_labels

    # # new fast train Without Grouper Additions
    # training_x = loader.training_x
    # training_y = loader.training_y

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
    testing_index = [i for i in range(len(loader.testing)) if loader.testing[i] not in new_pos] # why use all of loader.testing instead of just loader.grouper_testing

    testing = [loader.testing[i] for i in testing_index]
    testing_y = [loader.testing_y[i] for i in testing_index]

    log_json({'intent_mapping': loader.intent_dict, 'training_x': loader.training_x, 'training_y': loader.training_y, 'testing': testing, 'testing_y': testing_y, 'validation': loader.validation, 'validation_y': loader.validation_y, 'pruner_training_x': pruner_training_x, 'pruner_training_y': pruner_training_y}, rc.DATASET, i, "loader_dict.json")

    # new fast test
    testing = loader.testing
    testing_y = loader.testing_y

    # old merging: why is lf_testing and lf_testing_y brought up again if testing, testing_y already have them?
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


    pruned_lfs_index_tuple = pruner.prune_lfs_threshold_n_classes(max_lf=len(loader.intent_dict.values())*5, min_lf=len(loader.intent_dict.values()), threshold=rc.PRUNER_SCORE_THRESHOLD, main_iteration=i)

    pruner_dict = {}

    for item in (pruner.prune_intent_query, pruner.prune_emp_accuracy, pruner.prune_polarity, pruner.prune_coverage):
        pruned_lfs_index_low = item(type='low')
        pruner_dict[item.__name__ + '_low'] = pruned_lfs_index_low

        pruned_lfs_index_medium = item(type='medium')
        pruner_dict[item.__name__ + '_medium'] = pruned_lfs_index_medium

        pruned_lfs_index_high = item(type='high')
        pruner_dict[item.__name__ + '_high'] = pruned_lfs_index_high

    lfs_index_tuple_main = [list(set(pruned_lfs_index_tuple[0])), list(set(pruned_lfs_index_tuple[0]+pruned_lfs_index_tuple[1])), list(set(pruned_lfs_index_tuple[0]+pruned_lfs_index_tuple[1]+pruned_lfs_index_tuple[2]))]

    for m, pruned_lfs_index in enumerate(lfs_index_tuple_main):

        print("LF INDEX PRINTING")
        print(pruned_lfs_index)

        pruned_lfs = [lfs[index] for index in pruned_lfs_index]

        for n, item in enumerate(pruned_lfs_index):
            n += 1
            n_old = n
            n = f'{m}_{n}'
            
            output_dir = f"./logs/{rc.DATASET}{n}/run_{i}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            lf_summary_df.to_csv(f'./logs/{rc.DATASET}{n}/run_{i}/lf_summary_unpruned.csv', index=False)

            rc.OUTPUT_PATH = './results/' + rc.DATASET + str(n) + '/'

            if not pruned_lfs[:n_old]:
                print("pruned_lfs is empty")
            
            else:
                sApplier = SnorkelLFApplier(pruned_lfs[:n_old])
                lf_matrix = sApplier.apply(pruner_training_x)

            print('PRUNED', pruned_lfs[:n_old])

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
            lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in pruned_lfs[:n_old]]

            # Add number_of_queries as a new column in lf_analysis
            lf_summary_df['number_of_queries'] = lf_number_of_queries

            # Add class labels
            lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in pruned_lfs[:n_old]]
            lf_summary_df['class_labels'] = lf_class_labels

            lf_summary_json = lf_summary_df.to_json(orient='records')
            try:
                lf_summary_df.to_csv(f'./logs/{rc.DATASET}{n}/run_{i}/lf_summary.csv', index=False)
            except OSError:
                os.makedirs(f'./logs/{rc.DATASET}{n}/run_{i}', exist_ok = True)
                lf_summary_df.to_csv(f'./logs/{rc.DATASET}{n}/run_{i}/lf_summary.csv', index=False)

            # Log out intent dict
            output_dir = f"./logs/{rc.DATASET}{n}/run_{i}"
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

            log_json({"preds": preds.tolist()}, f"{rc.DATASET}{n}", i, "mlv_output_t.json")
            log_json({"accuracy": majority_voter.get_accuracy(testing_y, preds)}, f"{rc.DATASET}{n}", i, "mlv_accuracy_output_t.json")

            # DATA LABELLING ON TEST DATASET AND VALIDATION DATASET
            lf_matrix = sApplier.apply(testing + loader.validation)

            MLV = MajorityLabelVoter(len(loader.all_intents))
            preds = MLV.predict(lf_matrix)
            # Print percent of correct predictions
            print("Percentage of correct predictions: ", majority_voter.get_accuracy(testing_y + loader.validation_y, preds))

            # Log MLV output and accuracy
            log_json({"preds": preds.tolist()}, f"{rc.DATASET}{n}", i, "mlv_output.json")
            log_json({"accuracy": majority_voter.get_accuracy(testing_y + loader.validation_y, preds)}, f"{rc.DATASET}{n}", i, "mlv_accuracy_output.json")
            
            # RASA DATASET GENERATION AND RASA TRAINING
            lf_matrix = sApplier.apply(testing)

            MLV = MajorityLabelVoter(len(loader.all_intents))
            preds = MLV.predict(lf_matrix)

            generator.generate_training(testing, preds.tolist(), loader.intent_dict)
            generator.generate_testing(loader.validation, loader.validation_y, loader.intent_dict)

            # Baseline: Score without any modification to the data (Only use the original data)
            # Applied: Score with grouper and labelling functions
            results_dir = os.path.join(rc.OUTPUT_PATH, "run_" + str(i))
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            baseline_dir = os.path.join(results_dir, "baseline")
            Path(baseline_dir).mkdir(parents=True, exist_ok=True)

            # Train Rasa multiple times using inbuilt API
            shutil.copy("./data/output/baseline_nlu.yml", "./rasa/nlu.yml")
            shutil.copy("./data/output/baseline_nlu.yml", f"{baseline_dir}/nlu.yml")
            shutil.copy("./data/output/baseline_testing.yml", f"{baseline_dir}/testing.yml")
            shutil.copy("./data/output/baseline_testing.yml", "./rasa/testing.yml")

            # Multiple times train and test Rasa to get the average accuracy
            # Remove files inside the directory
            for file in os.listdir(results_dir):
                file_path = os.path.join(results_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

            shutil.copy("./data/output/nlu.yml", f"{results_dir}/nlu.yml")
            shutil.copy("./data/output/test_data.yml", f"{results_dir}/testing.yml")

            tt.train(domain_file_path="./rasa/domain.yml",
                    config_file_path="./rasa/config.yml",
                    training_file_path="./data/output/baseline_nlu.yml",
                    output_dir=baseline_dir)
            tt.test(baseline_dir, "./data/output/baseline_testing.yml", baseline_dir)

            tt.train(domain_file_path="./rasa/domain.yml",
                    config_file_path="./rasa/config.yml",
                    training_file_path="./data/output/nlu.yml",
                    output_dir=results_dir)
            tt.test(results_dir, "./data/output/test_data.yml", results_dir)

    # Go through the results and get the average accuracy, macro, micro and weighted scores
    # Get all directories in the output directory
    dirs = [d for d in os.listdir(rc.OUTPUT_PATH) if os.path.isdir(os.path.join(rc.OUTPUT_PATH, d))]
    for directory in dirs:
        # Get information from baseline and applied runs
        baseline_dir = os.path.join(rc.OUTPUT_PATH, directory, "baseline")
        applied_dir = os.path.join(rc.OUTPUT_PATH, directory)
        with open(os.path.join(baseline_dir, "intent_report.json"), "r") as f:
            current_baseline = json.load(f)
            baseline = av_sc.add_to_results(baseline, current_baseline)
        with open(os.path.join(applied_dir, "intent_report.json"), "r") as f:
            current_applied = json.load(f)
            applied = av_sc.add_to_results(applied, current_applied)

    baseline = av_sc.average(baseline, len(dirs))
    applied = av_sc.average(applied, len(dirs))
    with open(os.path.join(rc.OUTPUT_PATH, "baseline_results.json"), "w") as f:
        json.dump(baseline, f, sort_keys=True, indent=4)
    with open(os.path.join(rc.OUTPUT_PATH, "applied_results.json"), "w") as f:
        json.dump(applied, f, sort_keys=True, indent=4)
    sys.exit(0)
