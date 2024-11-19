import numpy as np


def predict(matrix) -> list:
    """
    Takes in a matrix of documents and labelling function votes and returns a list of the labels for each row.

    :param matrix: Matrix of documents with labelling functions as columns
    :return: List of labels for each document
    """
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    labels = []
    for i in range(rows):
        label = {'-1': 0}
        max_count = 0
        for j in range(columns):
            if matrix[i][j] > max_count:
                max_count = matrix[i][j]
                label = j
        labels.append(label)
    return labels


def get_accuracy(testing_y, predictions) -> float:
    """
    Calculates the accuracy of the predictions.

    :param testing_y: Actual labels
    :param predictions: Predicted labels
    :return: Accuracy of the predictions
    """
    correct = 0
    for i in range(len(testing_y)):
        if testing_y[i] == predictions[i]:
            correct += 1
    return correct / len(testing_y)
