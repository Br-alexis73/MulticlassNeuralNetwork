import numpy as np


def cross_entropy_loss(predicted_labels, true_labels):
    sample_number = true_labels.shape[0]

    # computation of the cross entropy
    log_likelihood = -np.log(predicted_labels)  # the logarithm of the predicted probs
    cross_entropy = true_labels * log_likelihood
    loss = np.sum(cross_entropy) / sample_number

    return loss  # average loss per sample


def mean_squared_error_loss(predicted_labels, true_labels):
    sample_number = true_labels.shape[0]
    return np.sum((predicted_labels - true_labels) ** 2) / sample_number
