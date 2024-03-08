import numpy as np
from activation_functions import relu, relu_prime, sigmoid, softmax
from loss_functions import cross_entropy_loss, mean_squared_error_loss


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

    def backpropagation(self, inputs, target_outputs):
        # Forward pass to get the activations and pre-activation values for each layer
        activations = [inputs]  # List to store activations for each layer
        zs = []  # List to store pre-activation values for each layer

        # Forward pass through hidden layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activations.append(relu(z))

        # Forward pass through the output layer
        final_z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(final_z)
        activations.append(softmax(final_z))

        # Backward pass
        gradients = {
            "weights": [np.zeros_like(w) for w in self.weights],
            "biases": [np.zeros_like(b) for b in self.biases]
        }

        # Calculate the gradient of the loss function with respect to the output of the network
        delta = self.loss_derivative(activations[-1],
                                     target_outputs)  # This needs to be implemented based on the loss function

        # Backpropagate the error through the output layer
        gradients["weights"][-1] = np.dot(activations[-2].T, delta)
        gradients["biases"][-1] = delta

        # Backpropagate through the hidden layers
        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            sp = relu_prime(z)  # Assuming relu is used in hidden layers; implement this derivative function
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            gradients["weights"][-l] = np.dot(activations[-l - 1].T, delta)
            gradients["biases"][-l] = delta

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
                loss = cross_entropy_loss(output, label)
                total_loss += loss
                gradients = self.backpropagation(input_data, label)
                self.update_weights(gradients, learning_rate)
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data)}")

    def predict(self, inputs):
        return self.feedforward(inputs)