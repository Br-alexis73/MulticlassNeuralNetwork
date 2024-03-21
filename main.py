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
    hidden_sizes = [5, 3]
    nn = MultiClassNeuralNetwork(input_size, hidden_sizes, output_size)

    x_train = train_data.drop('Species', axis=1).values
    y_train = pd.get_dummies(train_data['Species']).values
    x_test = test_data.drop('Species', axis=1).values
    y_test = pd.get_dummies(test_data['Species']).values

    epochs = 100
    learning_rate = 0.001
    method = "batch"

    # The train function should perform one epoch of training and return the loss and accuracy
    losses, train_accuracy, test_accuracy = nn.train(x_train, y_train, x_test, y_test, learning_rate, epochs, method=method)


    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training and testing accuracies
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(test_accuracy, label='Testing Accuracy')
    plt.title(f'Accuracy over Epochs (LR: {learning_rate})')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Training Loss')
    plt.title(f'Loss over Epochs (GD Method: {method})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    nn.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
