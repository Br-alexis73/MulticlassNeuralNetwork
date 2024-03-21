import numpy as np


def compute_accuracy(predictions, labels):
    """Compute accuracy of predictions."""
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy





