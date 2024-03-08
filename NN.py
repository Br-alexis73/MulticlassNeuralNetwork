import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)


class MultiClassNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros(hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]))
            self.biases.append(np.zeros(hidden_sizes[i]))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros(output_size))

    def feedforward(self, inputs):
        activations = inputs
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations = relu(np.dot(activations, w) + b)
        output = softmax(np.dot(activations, self.weights[-1]) + self.biases[-1])
        return output

    def cross_entropy_loss(self, output, target):
        # Ensure that output is 2-dimensional
        output = np.atleast_2d(output)

        # Calculate log_likelihood for correct class predictions
        log_likelihood = -np.log(output[np.arange(len(output)), target.argmax(axis=1)])
        loss = np.sum(log_likelihood) / len(log_likelihood)
        return loss

    def backpropagation(self, inputs, target_outputs):
        gradients = {
            "weights": [np.zeros_like(w) for w in self.weights],
            "biases": [np.zeros_like(b) for b in self.biases]
        }
        return gradients

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients["weights"][i]
            self.biases[i] -= learning_rate * gradients["biases"][i]

    def train(self, data, labels, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for i, (input_data, label) in enumerate(zip(data, labels)):
                output = self.feedforward(input_data)
                loss = self.cross_entropy_loss(output, label)
                total_loss += loss
                gradients = self.backpropagation(input_data, label)
                self.update_weights(gradients, learning_rate)
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data)}")

    def predict(self, inputs):
        return self.feedforward(inputs)