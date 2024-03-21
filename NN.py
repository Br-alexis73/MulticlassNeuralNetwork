import numpy as np
from activation_functions import relu, relu_prime, sigmoid, sigmoid_prime, softmax
from loss_functions import cross_entropy_loss, mean_squared_error_loss


class MultiClassNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, loss_function='cross_entropy_loss'):
        self.weights = []
        self.biases = []

        # Improved weight initialization (He initialization for ReLU activation)
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros(hidden_sizes[0]))


        #Hidden Layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * np.sqrt(2. / hidden_sizes[i - 1]))
            self.biases.append(np.zeros(hidden_sizes[i]))

        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2. / hidden_sizes[-1]))
        self.biases.append(np.zeros(output_size))

        if loss_function not in ['cross_entropy', 'mean_squared_error']:
            raise ValueError("Unsupported loss function")
        self.loss_function = loss_function

    def feedforward(self, inputs):
        # reshape input if it's 1D (i.e., one sample)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)

        activations = [inputs]
        z_values = []  # Store pre-activation values for backpropagation
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z_values.append(np.dot(activations[-1], w) + b)
            activations = relu(np.dot(activations, w) + b)
        output = softmax(np.dot(activations, self.weights[-1]) + self.biases[-1])
        return output

    def backpropagation(self, inputs, target_outputs):
        activations = self.feedforward(inputs)
        gradients = self._initialize_gradients()

        # Select the appropriate loss derivative function
        if self.loss_function == 'cross_entropy':
            delta = mean_squared_error_loss(activations[-1], target_outputs)
        else:
            raise ValueError("Unsupported loss function")

        gradients = self.backpropagation(delta, inputs, target_outputs)
        return gradients

    def _initialize_gradients(self):
        gradients = {
            "weights": [np.zeros_like(w) for w in self.weights],
            "biases": [np.zeros_like(b) for b in self.biases]
        }
        return gradients

    def update_weights(self, gradients, learning_rate):
        # Update weights and biases based on gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients["weights"][i]
            self.biases[i] -= learning_rate * gradients["biases"][i]

    def train(self, data, labels, epochs, learning_rate):
        labels = self.one_hot_encode(labels, num_classes=3)
        # Training loop over the specified number of epochs
        for epoch in range(epochs):
            total_loss = 0
            for input_data, label in zip(data, labels):
                output = self.feedforward(input_data)
                # Compute loss based on the specified loss function
                if self.loss_function == 'cross_entropy':
                    loss = cross_entropy_loss(output, label)
                else:
                    raise ValueError("Unsupported loss function")
                total_loss += loss

                gradients = self.backpropagation(input_data, label)
                self.update_weights(gradients, learning_rate)

            average_loss = total_loss / len(data)
            print(f"Epoch {epoch + 1}, Loss: {average_loss}")

    def one_hot_encode(self, labels, num_classes):
        one_hot_labels = np.zeros((len(labels), num_classes))
        one_hot_labels[np.arange(len(labels)), labels.astype(int)] = 1
        return one_hot_labels

    def predict(self, inputs):
        # Use the trained network for prediction
        return self.feedforward(inputs)
