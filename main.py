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



def main():
    train_data, test_data = preprocessing()
    print("\nTraining Data:")
    print(train_data.head())
    print("\nTesting Data:")
    print(test_data.head())

    input_size = train_data.shape[1] - 1
    output_size = len(np.unique(train_data['Species']))
    hidden_sizes = [8, 5, 7]

    x_train = train_data.drop('Species', axis=1).values
    y_train = pd.get_dummies(train_data['Species']).values
    x_test = test_data.drop('Species', axis=1).values
    y_test = pd.get_dummies(test_data['Species']).values

    loss_functions = ['cross_entropy', 'mse', 'mae']
    regularizations = [0.01]
    learning_rates = [0.01, 0.001]
    epochs = 100
    methods = ['sgd', 'mini-batch', 'batch']

    # Results storage
    results = []

    for loss_function in loss_functions:
        for regularization in regularizations:
            for learning_rate in learning_rates:
                for method in methods:
                    # Initialize the neural network with the current set of hyperparameters
                    nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size, loss_function=loss_function,
                                                 regularization_strength=regularization)

                    # Train the network and get training results
                    losses, train_accuracy, test_accuracy = nn.train(x_train, y_train, x_test, y_test, learning_rate,
                                                                     epochs, method=method)

                    # Store hyperparameters and the final accuracy or loss
                    results.append({
                        'loss_function': loss_function,
                        'regularization': regularization,
                        'learning_rate': learning_rate,
                        'method': method,
                        'final_train_accuracy': train_accuracy[-1],
                        'final_test_accuracy': test_accuracy[-1],
                        'final_loss': losses[-1]
                    })

    # Convert results to DataFrame for easy manipulation and plotting
    results_df = pd.DataFrame(results)

    sorted_results = results_df.sort_values('final_test_accuracy', ascending=False)

    # Select the best result
    best_result = sorted_results.iloc[0]

    # Initialize the best model with the best hyperparameters
    best_nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size,
                                      loss_function=best_result['loss_function'],
                                      regularization_strength=best_result['regularization'])

    # Train the best model to get the epoch-wise accuracies and losses
    losses, train_accuracy, test_accuracy = best_nn.train(x_train, y_train, x_test, y_test,
                                                          best_result['learning_rate'], epochs,
                                                          method=best_result['method'])

    # Now you can plot the epoch-wise accuracies and losses
    plt.figure(figsize=(14, 6))

    # Plot training and testing accuracies
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(test_accuracy, label='Testing Accuracy')
    plt.title(f'Best Accuracy over Epochs\n(LR: {best_result["learning_rate"]}, '
              f'Method: {best_result["method"]}, '
              f'Loss: {best_result["loss_function"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Training Loss')
    plt.title(f'Best Loss over Epochs\n(LR: {best_result["learning_rate"]}, '
              f'Method: {best_result["method"]}, '
              f'Loss: {best_result["loss_function"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    best_nn.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
