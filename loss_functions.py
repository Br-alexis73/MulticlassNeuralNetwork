import numpy as np


def cross_entropy_loss(predicted_labels, labels):
    print("Labels:", labels)
    print("Labels type:", type(labels))
    print("Labels shape:", labels.shape if isinstance(labels, np.ndarray) else "Not a numpy array")

    sample_number = labels.shape[0]

    # Add a small value to the predicted labels to avoid taking log of zero
    predicted_labels = np.clip(predicted_labels, 1e-10, 1. - 1e-10)

    # Computation of the cross entropy
    log_likelihood = -np.log(predicted_labels)
    cross_entropy = labels * log_likelihood
    loss = np.sum(cross_entropy) / sample_number

    return loss  # average loss per sample


def mean_absolute_error_loss(predicted_labels, true_labels):
    sample_number = true_labels.shape[0]
    return np.sum(np.abs(predicted_labels - true_labels)) / sample_number


def mean_squared_error_loss(predicted_labels, true_labels):
    sample_number = true_labels.shape[0]
    return np.sum((predicted_labels - true_labels) ** 2) / (2 * sample_number)

