import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from activations import relu, relu_derivative, softmax
from losses import cross_entropy_loss, mean_squared_error, mean_absolute_error
from metrics import compute_accuracy


class MultiClassNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, loss_function='cross_entropy', regularization_strength=0.01):
        # Initialize weights and biases for all layers
        self.regularization_strength = regularization_strength
        self.loss_function = loss_function
        self.weights = []
        self.biases = []

        """He initialization for ReLU activation"""
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros(hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]))
            self.biases.append(np.zeros(hidden_sizes[i]))

        # Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros(output_size))

    def forward(self, inputs):
        """Perform the forward pass."""
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

        return activations, z_values, predictions

    def compute_loss(self, predictions, labels):
        # Select the loss function
        if self.loss_function == 'cross_entropy':
            loss = cross_entropy_loss(predictions, labels)
        elif self.loss_function == 'mse':
            loss = mean_squared_error(predictions, labels)
        elif self.loss_function == 'mae':
            loss = mean_absolute_error(predictions, labels)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        # L2 regularization penalty
        l2_penalty = sum(np.sum(w ** 2) for w in self.weights)
        # Updated loss with L2 penalty
        loss += self.regularization_strength / 2 * l2_penalty
        # Accuracy computation remains unchanged
        accuracy = compute_accuracy(predictions, labels)
        return loss, accuracy

    def set_loss_function(self, new_loss_function):
        """Set a new loss function for the neural network."""
        self.loss_function = new_loss_function

    def set_regularization_strength(self, new_strength):
        self.regularization_strength = new_strength

    def compute_loss_and_gradients(self, inputs, labels):
        """Compute loss, accuracy, and gradients for backpropagation."""
        activations, z_values, predictions = self.forward(inputs)
        loss, accuracy = self.compute_loss(predictions, labels)
        gradients = self.backpropagation(activations, z_values, labels)
        return loss, accuracy, gradients

    def backpropagation(self, activations, z_values, labels):
        # Initialize gradients for weights and biases
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = activations[-1] - labels
        gradients_w[-1] = np.dot(activations[-2].T, delta) + self.regularization_strength * self.weights[-1]
        gradients_b[-1] = np.sum(delta, axis=0)

        # Propagate error backwards
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(delta, self.weights[-l + 1].T) * relu_derivative(z_values[-l])
            gradients_w[-l] = np.dot(activations[-l - 1].T, delta)
            gradients_b[-l] = np.sum(delta, axis=0)

            # Apply L2 regularization to the gradient for the weights
            gradients_w[-l] += self.regularization_strength * self.weights[-l]

        return gradients_w, gradients_b

    def update_parameters(self, gradients, learning_rate):
        gradients_w, gradients_b = gradients
        # Update weights and biases with gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train_epoch(self, inputs, labels, learning_rate):
        """Train for a single epoch and return loss and accuracy."""
        # Compute loss and gradients for one batch of data
        loss, accuracy, gradients = self.compute_loss_and_gradients(inputs, labels)
        # Update the network parameters
        self.update_parameters(gradients, learning_rate)
        return loss, accuracy

    def train(self, x_train, y_train, x_test, y_test, learning_rate, epochs, method='sgd', batch_size=32):
        """Run the training process over the full number of epochs."""
        losses, training_accuracies, testing_accuracies = [], [], []

        for epoch in range(1, epochs + 1):
            # Choose the gradient descent approach
            if method == 'sgd':
                # Implement SGD logic: Iterate over each example
                for i in range(len(x_train)):
                    loss, accuracy = self.train_epoch(x_train[i:i + 1], y_train[i:i + 1], learning_rate)
            elif method == 'batch':
                # Implement Batch GD logic: Use all data
                loss, accuracy = self.train_epoch(x_train, y_train, learning_rate)
            elif method == 'mini-batch':
                # Implement Mini-batch GD logic: Split data into batches
                for i in range(0, len(x_train), batch_size):
                    x_batch = x_train[i:i + batch_size]
                    y_batch = y_train[i:i + batch_size]
                    loss, accuracy = self.train_epoch(x_batch, y_batch, learning_rate)
            else:
                raise ValueError("Invalid method chosen. Use 'sgd', 'batch', or 'mini-batch'.")

            # Evaluation on the testing set
            _, _, predictions = self.forward(x_test)
            test_loss, test_accuracy = self.compute_loss(predictions, y_test)

            # Log and store metrics
            losses.append(loss)
            training_accuracies.append(accuracy)
            testing_accuracies.append(test_accuracy)
            if epoch % 100 == 0:
                print('\n')
                print(f'Optimiser: {method}, Learning rate: {learning_rate}')
                print(
                    f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {accuracy:.2f}, Testing Accuracy: {test_accuracy:.2f}")

        return losses, training_accuracies, testing_accuracies

    def predict(self, x):
        """Predict method that returns the predictions for the input x"""
        _, _, predictions = self.forward(x)
        return np.argmax(predictions, axis=1)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_true = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
