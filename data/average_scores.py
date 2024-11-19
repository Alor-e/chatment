def add_to_results(results, current_results):
    results["accuracy"] += current_results["accuracy"]
    results["macro avg"]["precision"] += current_results["macro avg"]["precision"]
    results["macro avg"]["recall"] += current_results["macro avg"]["recall"]
    results["macro avg"]["f1-score"] += current_results["macro avg"]["f1-score"]
    results["micro avg"]["precision"] += current_results["micro avg"]["precision"]
    results["micro avg"]["recall"] += current_results["micro avg"]["recall"]
    results["micro avg"]["f1-score"] += current_results["micro avg"]["f1-score"]
    results["weighted avg"]["precision"] += current_results["weighted avg"]["precision"]
    results["weighted avg"]["recall"] += current_results["weighted avg"]["recall"]
    results["weighted avg"]["f1-score"] += current_results["weighted avg"]["f1-score"]
    return results


def average(results, amount):
    results["accuracy"] /= amount
    results["macro avg"]["precision"] /= amount
    results["macro avg"]["recall"] /= amount
    results["macro avg"]["f1-score"] /= amount
    results["micro avg"]["precision"] /= amount
    results["micro avg"]["recall"] /= amount
    results["micro avg"]["f1-score"] /= amount
    results["weighted avg"]["precision"] /= amount
    results["weighted avg"]["recall"] /= amount
    results["weighted avg"]["f1-score"] /= amount
    return results


baseline = {"accuracy": 0,
            "macro avg":
                {"precision": 0,
                 "recall": 0,
                 "f1-score": 0
                 },
            "micro avg":
                {"precision": 0,
                 "recall": 0,
                 "f1-score": 0
                 },
            "weighted avg":
                {"precision": 0,
                 "recall": 0,
                 "f1-score": 0
                 }
            }
applied = {"accuracy": 0,
           "macro avg":
               {"precision": 0,
                "recall": 0,
                "f1-score": 0
                },
           "micro avg":
               {"precision": 0,
                "recall": 0,
                "f1-score": 0
                },
           "weighted avg":
               {"precision": 0,
                "recall": 0,
                "f1-score": 0
                }
           }
