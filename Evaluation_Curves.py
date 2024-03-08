import numpy as np
from Perceptron import NeuralNetwork
from Evaluation_Validation import calculate_accuracy  
import matplotlib.pyplot as plt  

# Function to plot learning curves
def plot_learning_curves(losses, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.plot(train_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

# Hyperparameter Tuning Helper Function
def hyperparameter_tuning(X_train, y_train, X_val, y_val, hidden_size, learning_rate_values, epochs_values):
    results = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for learning_rate in learning_rate_values:
        for epochs in epochs_values:
            model = NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[hidden_size], output_size=3)
            model.train(X_train, y_train.values, epochs=epochs, lr=learning_rate, X_val=X_val, y_val=y_val)

            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)

            train_accuracy = calculate_accuracy(y_train.values, train_predictions)
            val_accuracy = calculate_accuracy(y_val.values, val_predictions)

            results.append({
                'Learning Rate': learning_rate,
                'Epochs': epochs,
                'Train Accuracy': train_accuracy,
                'Validation Accuracy': val_accuracy
            })

            # Plot learning curves for the last training
            train_losses.append(model.train_losses)
            val_losses.append(model.val_losses)
            train_accuracies.append(model.train_accuracies)
            val_accuracies.append(model.val_accuracies)
            last_training_losses = model.losses
            plot_learning_curves(last_training_losses, train_losses, val_losses, train_accuracies, val_accuracies)

    return results

# Multi-Layer Neural Network Class
class MultiLayerNeuralNetwork(NeuralNetwork):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__(input_size, hidden_sizes, output_size)
        self.weights.append(np.random.rand(hidden_sizes[-1], 4))
        self.biases.append(np.zeros((1, 4)))

# Hyperparameter Tuning for Multi-Layer Neural Network
def multi_layer_hyperparameter_tuning(X_train, y_train, X_val, y_val, hidden_sizes, learning_rate_values, epochs_values):
    results = []
    for learning_rate in learning_rate_values:
        for epochs in epochs_values:
            model = MultiLayerNeuralNetwork(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=3)
            model.train(X_train, y_train.values, epochs=epochs, lr=learning_rate)

            val_predictions = model.predict(X_val)
            val_accuracy = calculate_accuracy(y_val.values, val_predictions)

            results.append({
                'Learning Rate': learning_rate,
                'Epochs': epochs,
                'Validation Accuracy': val_accuracy
            })

            # Plot learning curves for the last training
            last_multi_layer_training_losses = model.losses
            plot_learning_curves(last_multi_layer_training_losses)

    return results
