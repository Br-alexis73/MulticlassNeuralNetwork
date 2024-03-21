import numpy as np


def cross_entropy_loss(predictions, labels):
    """Categorical cross-entropy loss."""
    # Number of samples
    n_samples = labels.shape[0]
    # Clip predictions to avoid log(0) error
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    # Compute cross-entropy loss
    loss = -np.sum(labels * np.log(predictions)) / n_samples
    return loss


def mean_squared_error(y_true, y_pred):
    """Compute the mean squared error between true and predicted values"""
    return np.mean(np.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    """Compute the mean absolute error between true and predicted values"""
    return np.mean(np.abs(y_true - y_pred))
