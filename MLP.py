import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return np.where(x > 0, 1, 0)


def softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def cross_entropy_loss(predictions, labels):
    """Categorical cross-entropy loss."""
    # Number of samples
    n_samples = labels.shape[0]
    # Clip predictions to avoid log(0) error
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    # Compute cross-entropy loss
    loss = -np.sum(labels * np.log(predictions)) / n_samples
    return loss


def compute_accuracy(predictions, labels):
    """Compute accuracy of predictions."""
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy


class MultiClassNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []

        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros(hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]))
            self.biases.append(np.zeros(hidden_sizes[i]))

        # Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros(output_size))

    def compute_loss_and_gradients(self, inputs, labels):
        # Forward pass
        activations = [inputs]
        z_values = []  # Store pre-activation values for backpropagation

        # Pass through hidden layers with ReLU
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            activations.append(relu(z))

        # Output layer with Softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        predictions = softmax(z)
        activations.append(predictions)

        # Compute loss
        loss = cross_entropy_loss(predictions, labels)
        accuracy = compute_accuracy(predictions, labels)

        # Backward pass to compute gradients
        gradients = self.backpropagation(activations, z_values, labels)

        return loss, accuracy, gradients

    def backpropagation(self, activations, z_values, labels):
        # Initialize gradients for weights and biases
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = activations[-1] - labels
        gradients_w[-1] = np.dot(activations[-2].T, delta)
        gradients_b[-1] = np.sum(delta, axis=0)

        # Propagate error backwards
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(delta, self.weights[-l + 1].T) * relu_derivative(z_values[-l])
            gradients_w[-l] = np.dot(activations[-l - 1].T, delta)
            gradients_b[-l] = np.sum(delta, axis=0)

        return (gradients_w, gradients_b)

    def update_parameters(self, gradients, learning_rate):
        gradients_w, gradients_b = gradients
        # Update weights and biases with gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, training_data, labels, epochs, learning_rate):
        for epoch in range(epochs):
            loss, accuracy, gradients = self.compute_loss_and_gradients(training_data, labels)
            self.update_parameters(gradients, learning_rate)
            if epoch % 100 == 0:  # Print every 100 epochs
                print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

# Assuming you have training_data (inputs) and labels (one-hot encoded)
# nn.train(training_data, labels, epochs=1000, learning_rate=0.01)


# Generate synthetic data
np.random.seed(42)  # For reproducibility
num_samples_per_class = 100
input_size = 2  # 2D features
num_classes = 3  # 3 classes

# Features
X = np.vstack([
    np.random.randn(num_samples_per_class, input_size) + np.array([2, 2]),
    np.random.randn(num_samples_per_class, input_size) + np.array([0, -2]),
    np.random.randn(num_samples_per_class, input_size) + np.array([-2, 2])
])

# Labels (not one-hot encoded yet)
y = np.array([0] * num_samples_per_class + [1] * num_samples_per_class + [2] * num_samples_per_class)

# Convert labels to one-hot encoding
Y_one_hot = np.zeros((y.size, num_classes))
Y_one_hot[np.arange(y.size), y] = 1

# Split the data into training and test sets (here, we'll use all data for training for simplicity)
X_train = X
Y_train = Y_one_hot

# Initialize the neural network
hidden_sizes = [5, 4]  # Example: two hidden layers with 5 and 4 neurons
output_size = num_classes  # Matching the number of classes
nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size)

# Train the neural network
epochs = 1000
learning_rate = 0.01
nn.train(X_train, Y_train, epochs, learning_rate)
