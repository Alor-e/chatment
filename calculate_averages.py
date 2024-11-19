import json
import os

def calculate_averages():
    auc_scores = []
    accuracies = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(10):
        run_path = f'logs/msr_random/run_{i}'
        
        # Read AUC score
        with open(os.path.join(run_path, 'mlv_auc_score.json'), 'r') as f:
            auc_data = json.load(f)
            auc_scores.append(auc_data['auc_score'])
        
        # Read accuracy
        with open(os.path.join(run_path, 'mlv_accuracy_output.json'), 'r') as f:
            accuracy_data = json.load(f)
            accuracies.append(accuracy_data['accuracy'])
        
        # Read F1 score
        with open(os.path.join(run_path, 'mlv_f1_score.json'), 'r') as f:
            f1_data = json.load(f)
            f1_scores.append(f1_data['f1_score'])
        
        # Read precision score
        with open(os.path.join(run_path, 'mlv_precision_score.json'), 'r') as f:
            precision_data = json.load(f)
            precision_scores.append(precision_data['precision_score'])
        
        # Read recall score
        with open(os.path.join(run_path, 'mlv_recall_score.json'), 'r') as f:
            recall_data = json.load(f)
            recall_scores.append(recall_data['recall_score'])
    
    avg_auc = sum(auc_scores) / len(auc_scores)
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    print(f"Average AUC score: {avg_auc}")
    print(f"Average accuracy: {avg_accuracy}")
    print(f"Average F1 score: {avg_f1}")
    print(f"Average precision score: {avg_precision}")
    print(f"Average recall score: {avg_recall}")

if __name__ == "__main__":
    calculate_averages()