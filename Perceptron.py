import numpy as np
import random
from activation_functions import sigmoid, sigmoid_prime, relu, relu_prime, cross_entropy_loss, softmax


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', lr=0.01):
        # self.hidden_layer_input = hidden_layers
        self.hidden_layer_input = None
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = lr
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        # input to hidden
        self.weights = [np.random.randn(self.input_size, self.hidden_layers[0])]
        self.biases.append(np.zeros((1, self.output_size)))

    def feedforward(self, training_data):

        # hidden layers should have activation relu
        # then output layer should have activation softmax
        self.hidden_layer_input = np.dot(training_data, self.weights[0]) + self.biases[0]

        if self.activation == 'relu':
            hidden_layer_output = relu(self.hidden_layer_input)
        else:
            hidden_layer_output = softmax(self.hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights[1]) + self.biases[1]
        output_layer_predictions = softmax(output_layer_input)

        return output_layer_predictions

    def train(self, x, y, epochs, lr):
        self.losses = []
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        for epoch in range(epochs):
            hidden_layer_input = np.dot(x, self.weights[0]) + self.biases[0]
            if self.activation == 'relu':
                hidden_layer_output = np.maximum(0, hidden_layer_input)
            else:
                hidden_layer_output = softmax(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights[1]) + self.biases[1]
            output_layer_predictions = np.argmax(np.dot(hidden_layer_output, self.weights[1]) + self.biases[1], axis=1)

            output_error = (output_layer_predictions - y) * sigmoid_prime(output_layer_input)
            hidden_error = np.dot(output_error, self.weights[1].T) * self.relu_prime(hidden_layer_input)

            nabla_w_output = np.dot(hidden_layer_output.T, output_error)
            nabla_b_output = np.sum(output_error, axis=0, keepdims=True)

            nabla_w_hidden = np.dot(x.T, hidden_error)
            nabla_b_hidden = np.sum(hidden_error, axis=0, keepdims=True)

            self.weights[1] -= lr * nabla_w_output
            self.biases[1] -= lr * nabla_b_output
            self.weights[0] -= lr * nabla_w_hidden
            self.biases[0] -= lr * nabla_b_hidden

            loss = cross_entropy_loss(y, output_layer_predictions)
            self.losses.append(loss[0])  # Only append the loss value to the losses list
            self.train_losses.append(loss)

            # Calculate the training accuracy
            train_accuracy = self.accuracy_score(y, output_layer_predictions)
            self.train_accuracies.append(train_accuracy)

            # Calculate the validation accuracy
            # val_accuracy = self.compute_accuracy(X_val, y_val)
            # self.val_accuracies.append(val_accuracy)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Loss: {self.train_losses[-1]:.4f}, Train Accuracy: {self.train_accuracies[-1]:.4f}, Val Accuracy: {self.val_accuracies[-1]:.4f}")

    def predict(self, X):
        return self.feedforward(X)

    def accuracy_score(self, y_true, output_layer_predictions):
        # Calculate the number of correct predictions
        num_correct = np.sum(y_true == output_layer_predictions)

        # Calculate the total number of predictions
        num_total = len(y_true)

        # Calculate the accuracy as the number of correct predictions divided by the total number of predictions
        accuracy = num_correct / num_total

        return accuracy
