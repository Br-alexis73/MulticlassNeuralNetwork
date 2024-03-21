import numpy as np
from Perceptron import NeuralNetwork
from Evaluation_Validation import calculate_accuracy
import pandas as pd


# Cross-Validation Helper Function
def cross_validation(X, y, num_folds, hidden_size, learning_rate, epochs):
    fold_size = len(X) // num_folds
    results = []

    for fold in range(num_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_val_fold = X[start:end]
        y_val_fold = y.iloc[start:end]

        X_train_fold = np.concatenate([X[:start], X[end:]])
        y_train_fold = pd.concat([y.iloc[:start], y.iloc[end:]])

        model = NeuralNetwork(input_size=X_train_fold.shape[1], hidden_sizes=[hidden_size], output_size=3)
        model.train(X_train_fold, y_train_fold.values, epochs=epochs, lr=learning_rate)

        val_predictions = model.predict(X_val_fold)
        val_accuracy = calculate_accuracy(y_val_fold.values, val_predictions)

        results.append(val_accuracy)

    return np.mean(results)
