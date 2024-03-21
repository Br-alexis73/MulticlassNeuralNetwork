import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from model import MultiClassNeuralNetwork


def preprocessing():
    # Load the dataset
    hawks_raw = pd.read_csv('Hawks.csv')
    hawks_data = hawks_raw.drop(["BandNumber", "Sex", "Age"], axis=1)

    hawks_data = hawks_data.dropna()
    hawks_data = pd.DataFrame(hawks_data)

    # Define the label encoder
    encoder = LabelEncoder()
    hawks_data['Species'] = encoder.fit_transform(hawks_data['Species'])

    # MinMax scaling
    scaler_minmax = MinMaxScaler()
    # Select the columns you want to scale
    columns_to_scale = ['Wing', 'Weight', 'Culmen', 'Hallux', 'Tail']

    # Apply the scaler to the DataFrame
    hawks_data[columns_to_scale] = scaler_minmax.fit_transform(hawks_data[columns_to_scale])

    # Shuffle the dataset
    hawks_data = shuffle(hawks_data)

    # Split the dataset
    train_hawk, test_hawk = train_test_split(hawks_data, test_size=0.3, random_state=233)

    return train_hawk, test_hawk


def train_model(x_train, y_train, x_test, y_test, hidden_sizes, loss_function, regularization, learning_rate, epochs, method):
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size, loss_function, regularization)
    losses, train_accuracy, test_accuracy = nn.train(x_train, y_train, x_test, y_test, learning_rate, epochs, method)
    return nn, losses, train_accuracy, test_accuracy


def manual_tuning(x_train, y_train, x_test, y_test):
    hidden_sizes = [8, 5, 7]
    nn, losses, train_accuracy, test_accuracy = train_model(x_train, y_train, x_test, y_test, hidden_sizes,
                                                            'cross_entropy', 0.01, 0.01, 100, 'sgd')
    # Further actions based on the trained model and its performance
    print("Manual tuning done. Model trained.")


def automatic_tuning(x_train, y_train, x_test, y_test):
    best_accuracy = 0
    best_params = {}
    best_losses = None
    best_train_accuracy = None
    best_test_accuracy = None
    loss_functions = ['cross_entropy', 'mse', 'mae']
    regularizations = [0.01]
    learning_rates = [0.01, 0.001]
    epochs = 100
    methods = ['sgd', 'mini-batch', 'batch']
    hidden_sizes = [8, 3]

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


def main():
    train_data, test_data = preprocessing()
    print("\nTraining Data:")
    print(train_data.head())
    print("\nTesting Data:")
    print(test_data.head())

    x_train = train_data.drop('Species', axis=1).values
    y_train = pd.get_dummies(train_data['Species']).values
    x_test = test_data.drop('Species', axis=1).values
    y_test = pd.get_dummies(test_data['Species']).values

    # manual_tuning(x_train, y_train, x_test, y_test)
    nn, losses, train_accuracy, test_accuracy = automatic_tuning(x_train, y_train, x_test, y_test)

    # After the model is trained, we evaluate it on the test set
    nn.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
