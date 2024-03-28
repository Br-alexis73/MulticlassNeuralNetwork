import matplotlib.pyplot as plt
from model import MultiClassNeuralNetwork


def train_model(x_train, y_train, x_test, y_test, hidden_sizes, loss_function, regularization, learning_rate, epochs, method):
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size, loss_function, regularization)
    losses, train_accuracy, test_accuracy = nn.train(x_train, y_train, x_test, y_test, learning_rate, epochs, method)
    return nn, losses, train_accuracy, test_accuracy


def manual_tuning(x_train, y_train, x_test, y_test):
    hidden_sizes = list(map(int, input("Enter hidden layer sizes separated by space (e.g., 8 5 7): ").split()))
    learning_rate = float(input("Enter learning rate (e.g., 0.01): "))
    epochs = int(input("Enter number of epochs (e.g., 100): "))
    optimizer_method = input("Enter optimizer method (sgd or mini-batch): ")
    loss_function = input("Enter loss function (cross_entropy or mse): ")
    regularization = float(input("Enter regularization strength (e.g., 0.01): "))

    # Train model with user-defined parameters
    nn, losses, train_accuracy, test_accuracy = train_model(x_train, y_train, x_test, y_test, hidden_sizes,
                                                            loss_function, regularization, learning_rate, epochs,
                                                            optimizer_method)

    # Prepare parameters dictionary for plotting
    params = {
        'learning_rate': learning_rate,
        'method': optimizer_method,
        'loss_function': loss_function
    }

    # Plot results using the provided function
    plot_results(params, losses, train_accuracy, test_accuracy, epochs)
    return nn, losses, train_accuracy, test_accuracy


def automatic_tuning(x_train, y_train, x_test, y_test):
    best_accuracy = 0
    best_params = {}
    best_losses = None
    best_train_accuracy = None
    best_test_accuracy = None
    loss_functions = ['cross_entropy', 'mse', 'mae']
    regularizations = [0.1, 0.01, 0.001]
    learning_rates = [0.1, 0.01, 0.001]
    epochs = 100
    methods = ['sgd', 'mini-batch']
    hidden_sizes_input = input("Enter hidden layer sizes separated by space (e.g., 8 5 7): ")
    hidden_sizes = list(map(int, hidden_sizes_input.split()))

    for loss_function in loss_functions:
        for regularization in regularizations:
            for learning_rate in learning_rates:
                for method in methods:
                    nn, losses, train_accuracy, test_accuracy = train_model(
                        x_train, y_train, x_test, y_test, hidden_sizes, loss_function,
                        regularization, learning_rate, epochs, method)

                    current_accuracy = max(test_accuracy)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_losses = losses
                        best_train_accuracy = train_accuracy
                        best_test_accuracy = test_accuracy
                        best_params = {
                            'hidden_sizes': hidden_sizes,
                            'loss_function': loss_function,
                            'regularization': regularization,
                            'learning_rate': learning_rate,
                            'method': method,
                            'accuracy': best_accuracy
                        }

    print(f"Best Parameters: {best_params}")
    plot_results(best_params, best_losses, best_train_accuracy, best_test_accuracy, epochs)
    return nn, best_losses, best_train_accuracy, best_test_accuracy


def plot_results(params, losses, train_acc, test_acc, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    title_suffix = f"(LR: {params['learning_rate']}, Method: {params['method']}, Loss: {params['loss_function']})"

    ax1.plot(range(epochs), train_acc, label='Training Accuracy')
    ax1.plot(range(epochs), test_acc, label='Testing Accuracy')
    ax1.set_title(f'Best Accuracy over Epochs\n{title_suffix}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(range(epochs), losses, label='Training Loss')
    ax2.set_title(f'Best Loss over Epochs\n{title_suffix}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    plt.show()

