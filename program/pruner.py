import numpy as np
from pandas import DataFrame
import program.run_config as rc

from snorkel.labeling.apply.core import LFApplier as SnorkelLFApplier
from snorkel.labeling.analysis import LFAnalysis

from sklearn.metrics import f1_score, roc_auc_score
from typing import List
import math


def calculate_average_auc_safe(gold_labels: List[int], predicted_labels: List[int]) -> float:
    """
    Calculate the average AUC for a multi-class classification problem, with handling for empty unique labels.
    
    Parameters:
    - gold_labels (List[int]): A list of the actual labels.
    - predicted_labels (List[int]): A list of the predicted labels.
    
    Returns:
    - float: The average AUC or 0 if unique_labels is empty.
    """
    sum_auc = 0.0
    unique_labels = np.unique(gold_labels)
    
    if len(unique_labels) == 0:
        return 0.0
    
    for label in unique_labels:
        binary_gold_labels = [1 if x == label else 0 for x in gold_labels]
        binary_predicted = [1 if x == label else 0 for x in predicted_labels]
        
        auc = roc_auc_score(binary_gold_labels, binary_predicted)
        sum_auc += auc
    
    average_auc = sum_auc / len(unique_labels)
    return average_auc


class Pruner:
    """
    Prunes labelling functions from the list of labelling functions depending on how well they perform.
    """

    def __init__(self, lfs: list = None, pruner_training_x: list = None, pruner_training_y: list = None, number_of_classes: int = None):
        self.lfs = lfs
        self.pruner_training_x = pruner_training_x
        self.pruner_training_y = pruner_training_y
        self.number_of_classes = number_of_classes


    def prune_lfs(self, n: int):
        """Prune labelling functions to retain only top 'n' based on accuracy and coverage."""

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)


        lf_analysis['Score'] = lf_analysis['Emp. Acc.']

        # Sort the labeling functions by the combined score in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='Score', ascending=False)

        # Select top 'n' labeling functions
        top_lfs = lf_analysis_sorted.head(n)
        return top_lfs.index.values  # Return indices of top 'n' LFs
    

    def prune_lfs_per_type(self, n: int):
        """Prune labelling functions to retain only top 'n' per type based on accuracy and coverage."""

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # define lf_types and lf_number_of_queries based on class names
        lf_types = [type(lf).__name__ for lf in self.lfs]
        print("LF TYPE LIST")
        print(lf_types)
        lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in self.lfs]

        # Calculate a score for each labeling function
        lf_analysis['Score'] = lf_analysis['Emp. Acc.']

        # Collect scores and types for each LF
        lf_scores = lf_analysis['Score'].tolist()

        # Group LFs by type, score and number of queries
        lfs_by_type = {}
        for i, (lf_type, score, num_queries) in enumerate(zip(lf_types, lf_scores, lf_number_of_queries)):
            if lf_type in ["RandomForestLabeller", "DecisionTreeLabeller", "KNeighborsLabeller", 
                            "LogisticRegressionLabeller", "SVMLabeller"]:
                lf_type = "MachineLabeller"
            if lf_type not in lfs_by_type:
                lfs_by_type[lf_type] = []
            lfs_by_type[lf_type].append((i, score, num_queries))

        # Select top 'n' LFs for each type
        top_lfs = []
        for lf_type in lfs_by_type:
            print('LF_TYPE', lf_type)
            print(lfs_by_type[lf_type])
            lfs_by_type[lf_type].sort(key=lambda x: (x[1], x[2]), reverse=True)
            top_lfs.extend([lf[0] for lf in lfs_by_type[lf_type][:n]])

        return top_lfs  # Return indices of top 'n' LFs per type


    def prune_lfs_threshold(self, threshold: float = 0.7):
        """Prune labelling functions to retain only top 'n' based on accuracy and coverage."""

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # lf score is emperically accuracy
        lf_analysis['Score'] = lf_analysis['Emp. Acc.']

        # filter by the threshold
        lf_analysis = lf_analysis[lf_analysis['Emp. Acc.'] >= threshold]

        # Sort the labeling functions by the combined score in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='Score', ascending=False)

        # Select top 'n' labeling functions
        top_lfs = lf_analysis_sorted

        print("TOP LFs LENGTH")
        print(len(top_lfs))

        return top_lfs.index.values  # Return indices of top 'n' LFs
    

    def prune_lfs_threshold_n(self, max_lf: int = None, min_lf: int = None, threshold: float = 0.7):
        """Prune labelling functions to retain only top 'n' based on accuracy and coverage."""

        if min_lf == None:
            min_lf = self.number_of_classes
        
        if max_lf == None:
            max_lf = self.number_of_classes * 5 # 5 is the max multiplier

        if min_lf > max_lf:
            max_lf = min_lf

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # fetch the number of queries per lf
        lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in self.lfs]

        # Add number_of_queries as a new column in lf_analysis
        lf_analysis['number_of_queries'] = lf_number_of_queries

        # lf score is emperically accuracy
        lf_analysis['Score'] = lf_analysis['Emp. Acc.']

        if len(lf_analysis[lf_analysis['Emp. Acc.'] >= threshold]) < min_lf:
            # filter by the threshold and include 0 coverages
            lf_analysis = lf_analysis[(lf_analysis['Emp. Acc.'] >= threshold) | (lf_analysis['Coverage'] == 0)]

            # Sort the labeling functions by the combined score and number_of_queries in descending order
            lf_analysis_sorted = lf_analysis.sort_values(by=['Score', 'number_of_queries'], ascending=[False, False])

            # Select top 'n' labeling functions
            top_lfs = lf_analysis_sorted.head(min_lf)
        else:
            # filter by the threshold
            lf_analysis = lf_analysis[lf_analysis['Emp. Acc.'] >= threshold]

            # Sort the labeling functions by the combined score and number_of_queries in descending order
            lf_analysis_sorted = lf_analysis.sort_values(by=['Score', 'number_of_queries'], ascending=[False, False])

            # Select top 'n' labeling functions
            top_lfs = lf_analysis_sorted.head(max_lf)

        return top_lfs.index.values  # Return indices of top 'n' LFs
    

    def prune_lfs_threshold_n_classes(self, main_iteration: int, max_lf: int = None, min_lf: int = None, threshold: float = 0.7):
        """Prune labelling functions to retain only top 'n' based on accuracy and coverage."""

        if min_lf == None:
            min_lf = self.number_of_classes
        
        if max_lf == None:
            max_lf = self.number_of_classes * 5 # 5 is the max multiplier

        if min_lf > max_lf:
            max_lf = min_lf

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        # Calculating F1 scores and AUC for each LF
        f1_scores = []
        # auc_scores = []
        for i in range(lf_matrix.shape[1]):
            predicted_labels = lf_matrix[:, i]
            mask = predicted_labels != -1
            f1 = f1_score(np.array(self.pruner_training_y)[mask], predicted_labels[mask], average='weighted')
            # auc = calculate_average_auc_safe(np.array(self.pruner_training_y)[mask], predicted_labels[mask])
            
            f1_scores.append(f1)
            # auc_scores.append(auc)

        # Getting the LF analysis summary
        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Adding F1 and AUC scores to the DataFrame
        lf_analysis['F1'] = f1_scores
        # lf_analysis['AUC'] = auc_scores

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        lf_types = [type(lf).__name__ for lf in self.lfs]
        lf_analysis['lf_types'] = lf_types

        lf_types_groups = []
        for lf_type in lf_types:
            if lf_type in ["RandomForestLabeller", "DecisionTreeLabeller", "KNeighborsLabeller", 
                                "LogisticRegressionLabeller", "SVMLabeller"]:
                lf_type = "MachineLabeller"
            lf_types_groups.append(lf_type)

        lf_analysis['lf_types_groups'] = lf_types_groups


        # fetch the number of queries per lf
        lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in self.lfs]

        # Add number_of_queries as a new column in lf_analysis
        lf_analysis['number_of_queries'] = lf_number_of_queries

        # Add class labels
        lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in self.lfs]
        lf_analysis['class_labels'] = lf_class_labels

        # lf score is emperically accuracy
        lf_analysis['Score'] = lf_analysis['F1']

        # Create a new DataFrame that's a copy of lf_analysis
        lf_analysis_copy = lf_analysis.copy()

        # Remove the rows where 'class_labels' is None
        lf_analysis_copy = lf_analysis_copy[lf_analysis_copy['class_labels'].notna()]

        # Sort by score and number of queries per class label
        lf_analysis_copy_sorted = lf_analysis_copy.sort_values(by=['Score', 'number_of_queries'], ascending=[False, False])

        # Get the original indices of the first row for each class label
        # top_lf_class_indices_with_old_index = lf_analysis_copy_sorted.groupby('class_labels').apply(lambda x: x.index[0])

        # FOR LOGS NOT USED
        top_lf_class_indices_with_old_index = lf_analysis_copy_sorted.groupby('class_labels').apply(
            lambda x: None if x.name == -1 else x.index[0]
        )

        # Remove None values
        top_lf_class_indices_with_old_index = top_lf_class_indices_with_old_index.dropna()

        lf_analysis_copy_sorted.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_f1.csv', index=False)

        # # Create a new dataframe that only contains the rows with the indices in top_lf_class_indices_with_old_index
        selected_rows_df = lf_analysis_copy.loc[top_lf_class_indices_with_old_index]
        selected_rows_df.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_aa.csv', index=False)
        
        # top_lf_class_indices = selected_rows_df.index
        # FOR LOGS NOT USED ENDS HERE

        # NOW USED INSTEAD
        # Convert 'Polarity' to list of integers
        def convert_to_int_list(x):
            if isinstance(x, str):
                return [int(i) for i in x.strip('[]').split(',') if i]
            elif isinstance(x, list):
                return [int(i) for i in x]
            else:
                return x  # or however you want to handle other types

        lf_analysis_copy_sorted['Polarity'] = lf_analysis_copy_sorted['Polarity'].apply(convert_to_int_list)

        # lf_analysis_copy_sorted['Polarity'] = lf_analysis_copy_sorted['Polarity'].apply(lambda x: [int(i) for i in x.strip('[]').split(',') if i])

        # Explode the DataFrame on 'Polarity' to handle cases with multiple values in 'Polarity'
        df_exploded = lf_analysis_copy_sorted.explode('Polarity', ignore_index=False)

        # Filter out rows where 'Polarity' is empty
        df_exploded = df_exploded[df_exploded['Polarity'].notna()]

        # Group by 'Polarity' and pick the first occurrence of each unique value
        grouped_df = df_exploded.groupby('Polarity').apply(lambda x: x.iloc[0]).reset_index(drop=True)

        # Show the resulting DataFrame
        grouped_df.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_aa2.csv', index=False)
        # TEST

        # Remove the rows with top_lf_indices from the original dataframe
        # lf_analysis = lf_analysis.drop(top_lf_class_indices) 

        lf_analysis.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_bb.csv', index=False)

        # Filter by the threshold and number_of_queries
        lf_analysis_filtered = lf_analysis[(lf_analysis['Score'] >= threshold) & (lf_analysis['number_of_queries'] >= 2)]

        lf_analysis_filtered.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_cc.csv', index=False)

        # Sort the labeling functions by the combined score and number_of_queries in descending order
        lf_analysis_filtered_sorted = lf_analysis_filtered.sort_values(by=['Score', 'number_of_queries'], ascending=[False, False])
        lf_analysis_filtered.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_dd.csv', index=False)

        # Assuming lf_types_groups is a list
        unique_items = list(set(lf_types_groups))

        # Check if all unique items are present in the "lf_types_groups" column of grouped_df
        selected_rows_check = all(item in grouped_df['lf_types_groups'].values for item in unique_items)

        # Check if all unique items are present in the "lf_types_groups" column of lf_analysis_filtered
        lf_analysis_filtered_check = all(item in lf_analysis_filtered['lf_types_groups'].values for item in unique_items)

        # List to store the indexes of the first rows
        first_row_indexes = []

        # Sort the labeling functions by the combined score and number_of_queries in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by=['Score', 'number_of_queries'], ascending=[False, False])

        # If any unique item was not found in both DataFrames
        if not selected_rows_check and not lf_analysis_filtered_check:
            print("ALL COVERED")
            # Find the values that were not found
            not_found_values = [item for item in unique_items if item not in grouped_df['lf_types_groups'].values and item not in lf_analysis_filtered['lf_types_groups'].values]

            # Retrieve the first row and its index for each not found value from lf_analysis_sorted
            for value in not_found_values:
                first_row_series = lf_analysis_sorted[lf_analysis_sorted['lf_types_groups'] == value].iloc[0]
                first_row_index = first_row_series['j']

                # Append the index to the list
                first_row_indexes.append(first_row_index)

        # The list first_row_indexes now contains the indexes of the first rows for each not-found value
        print("IMPORTANT: Indexes of first rows:", first_row_indexes)

        top_lfs = lf_analysis_filtered_sorted

        top_lf_class_indices_list = grouped_df['j'].tolist()
        top_lfs_indices_list = top_lfs['j'].tolist()

        # Combine both lists
        combined_indices = top_lf_class_indices_list + top_lfs_indices_list + first_row_indexes
        combined_indices = list(set(combined_indices))

        print("Combined Indices:", combined_indices)
        print("LF Analysis Shape:", lf_analysis.shape)

        # Use iloc to select the valid rows
        # filtered_lf_analysis = lf_analysis.iloc[valid_combined_indices]
        filtered_lf_analysis = lf_analysis_copy[lf_analysis_copy['j'].isin(combined_indices)]

        # Save the filtered DataFrame to a CSV file
        filtered_lf_analysis.to_csv(f'./logs/{rc.DATASET}/run_{main_iteration}/lf_summary_enriched.csv', index=False)

        return combined_indices


    @staticmethod
    def lf_rank_selector_naive(array: list, n: int, type: str):
        if type == "medium":
            if len(array) < n:
                return array

            start_index = len(array) // 2 - n // 2
            end_index = start_index + n
            return array[start_index:end_index]
        
        elif type == "high":
            return array[:n]
        
        elif type == "low":
            return array[-n:]

    @staticmethod        
    def lf_rank_selector(array: list, n: int, type: str, center_index=None):
        if type == "medium":
            if len(array) < n:
                return array

            start_index = max(0, center_index - n // 2)
            end_index = min(len(array), start_index + n)

            return array[start_index:end_index]
        
        elif type == "high":
            return array[:n]
        
        elif type == "low":
            return array[-n:]

    
    def prune_intent_query(self, type="medium", n=20):
        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # fetch the number of queries per lf
        lf_number_of_queries = [lf.number_of_queries if hasattr(lf, 'number_of_queries') else 0 for lf in self.lfs]

        # Add number_of_queries as a new column in lf_analysis
        lf_analysis['number_of_queries'] = lf_number_of_queries

        # Sort the labeling functions by the number of queries in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='number_of_queries', ascending=False)

        # Recalculate the number of queries per lf after sorting
        lf_number_of_queries_sorted = lf_analysis_sorted['number_of_queries'].tolist()

        mean_value = sum(lf_number_of_queries_sorted) / len(lf_number_of_queries_sorted)
        closest_to_mean = min(lf_number_of_queries_sorted, key=lambda x: abs(x - mean_value))
        center_index = lf_number_of_queries_sorted.index(closest_to_mean)

        lfs_index = Pruner.lf_rank_selector(lf_analysis_sorted.index.values, n=n, type=type, center_index=center_index)
        return lfs_index


    def prune_coverage(self, type = "medium", n = 20):

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # Sort the labeling functions by the number of queries in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='Coverage', ascending=False)

        # Recalculate the number of queries per lf after sorting
        lf_coverage_sorted = lf_analysis_sorted['Coverage'].tolist()

        mean_value = sum(lf_coverage_sorted) / len(lf_coverage_sorted)
        closest_to_mean = min(lf_coverage_sorted, key=lambda x: abs(x - mean_value))
        center_index = lf_coverage_sorted.index(closest_to_mean)

        lfs_index = Pruner.lf_rank_selector(lf_analysis_sorted.index.values, n=n, type=type, center_index=center_index)
        return lfs_index
    

    def prune_emp_accuracy(self, type = "medium", n = 20):

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # Sort the labeling functions by the number of queries in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='Emp. Acc.', ascending=False)

        # Recalculate the number of queries per lf after sorting
        lf_emp_accuracy_sorted = lf_analysis_sorted['Emp. Acc.'].tolist()

        mean_value = sum(lf_emp_accuracy_sorted) / len(lf_emp_accuracy_sorted)
        closest_to_mean = min(lf_emp_accuracy_sorted, key=lambda x: abs(x - mean_value))
        center_index = lf_emp_accuracy_sorted.index(closest_to_mean)

        lfs_index = Pruner.lf_rank_selector(lf_analysis_sorted.index.values, n=n, type=type, center_index=center_index)
        return lfs_index
    

    def prune_f1(self, type = "medium", n=20):

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)
        # Calculating F1 scores and AUC for each LF
        f1_scores = []
        # auc_scores = []
        for i in range(lf_matrix.shape[1]):
            predicted_labels = lf_matrix[:, i]
            mask = predicted_labels != -1
            f1 = f1_score(np.array(self.pruner_training_y)[mask], predicted_labels[mask], average='weighted')
            # auc = calculate_average_auc_safe(np.array(self.pruner_training_y)[mask], predicted_labels[mask])
            
            f1_scores.append(f1)
            # auc_scores.append(auc)

        # Getting the LF analysis summary
        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Adding F1 and AUC scores to the DataFrame
        lf_analysis['F1'] = f1_scores

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # Sort the labeling functions by the number of queries in descending order
        lf_analysis_sorted = lf_analysis.sort_values(by='Emp. Acc.', ascending=False)

        # Recalculate the number of queries per lf after sorting
        lf_emp_accuracy_sorted = lf_analysis_sorted['Emp. Acc.'].tolist()

        mean_value = sum(lf_emp_accuracy_sorted) / len(lf_emp_accuracy_sorted)
        closest_to_mean = min(lf_emp_accuracy_sorted, key=lambda x: abs(x - mean_value))
        center_index = lf_emp_accuracy_sorted.index(closest_to_mean)

        lfs_index = Pruner.lf_rank_selector(lf_analysis_sorted.index.values, n=n, type=type, center_index=center_index)
        return lfs_index
    

    def prune_polarity(self, type = "medium", n = 20):

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # Assuming 'lf_analysis' is your DataFrame and 'Polarity' is the column with the lists
        lf_analysis['PolarityLength'] = lf_analysis['Polarity'].apply(len)

        # Now you can sort by 'PolarityLength'
        lf_analysis_sorted = lf_analysis.sort_values(by='PolarityLength', ascending=False)

        # Recalculate the number of queries per lf after sorting
        lf_polarity_sorted = lf_analysis_sorted['PolarityLength'].tolist()

        mean_value = sum(lf_polarity_sorted) / len(lf_polarity_sorted)
        closest_to_mean = min(lf_polarity_sorted, key=lambda x: abs(x - mean_value))
        center_index = lf_polarity_sorted.index(closest_to_mean)

        lfs_index = Pruner.lf_rank_selector(lf_analysis_sorted.index.values, n=n, type=type, center_index=center_index)
        return lfs_index
    

    def prune_quantity(self, n = 2000):

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        print(lf_analysis.head(n))

        print(lf_analysis.head(n).index.values)

        lfs_index = lf_analysis.head(n).index.values

        return lfs_index
    

    def prune_lfs_n_per_intent(self, n: int):
        """Prune labelling functions to retain only top 'n' based on accuracy and coverage."""

        sApplier = SnorkelLFApplier(self.lfs)
        lf_matrix = sApplier.apply(self.pruner_training_x)

        lf_analysis = LFAnalysis(L=lf_matrix, lfs=self.lfs).lf_summary(Y=np.array(self.pruner_training_y))

        # Reset the index of the dataframe
        lf_analysis.reset_index(drop=True, inplace=True)

        # Add class labels
        lf_class_labels = [lf.label if hasattr(lf, 'label') else None for lf in self.lfs]
        lf_analysis['class_labels'] = lf_class_labels

        # Create a new DataFrame that's a copy of lf_analysis
        lf_analysis_copy = lf_analysis.copy()

        # Remove the rows where 'class_labels' is None
        lf_analysis_copy = lf_analysis_copy[lf_analysis_copy['class_labels'].notna()]

        top_lf_class_indices_with_old_index = lf_analysis_copy.groupby('class_labels').apply(lambda x: list(x.index[:n])).explode()
        print(top_lf_class_indices_with_old_index)
        return top_lf_class_indices_with_old_index.to_list()
