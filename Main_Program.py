import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from NN import MultiClassNeuralNetwork
from loss_functions import cross_entropy_loss
from Evaluation_Validation import calculate_accuracy, plot_learning_curves
from Evaluation_Cross_Validation import cross_validation


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
    print("\\nTraining Data:")
    print(train_data.head())  # Show only the head for brevity
    print("\\nTesting Data:")
    print(test_data.head())  # Show only the head for brevity

    # Instantiate the neural network
    input_size = train_data.shape[1] - 1  # Subtract 1 for the target column
    output_size = len(np.unique(train_data['Species']))  # Number of species
    hidden_layers = [5, 3]  # Example: One hidden layer with 5 neurons
    nn = MultiClassNeuralNetwork(input_size, hidden_layers, output_size)

    # Prepare training data
    x_train = train_data.drop('Species', axis=1).values
    y_train = train_data['Species'].values

    # Prepare testing data
    x_test = test_data.drop('Species', axis=1).values
    y_test = test_data['Species'].values

    epochs = 100  # Number of times to loop through the entire dataset
    learning_rate = 0.01  # Learning rate for the optimizer

    # Lists to store metrics for plotting
    training_accuracies = []
    testing_accuracies = []
    losses = []

    for epoch in range(epochs):
        # Forward propagation
        predictions = nn.train(x_train, y_train, epochs, learning_rate)

        # Calculate loss
        loss = (y_train, predictions)
        losses.append(loss)

        # Backpropagation
        gradients = nn.backpropagation(y_train, predictions)

        # Update weights
        nn.update_weights(gradients, learning_rate)

        # Evaluate and store accuracies every 10 epochs, for example
        if epoch % 10 == 0:
            train_accuracy = calculate_accuracy(y_train, nn.feedforward(x_train))
            test_accuracy = calculate_accuracy(y_test, nn.feedforward(x_test))
            training_accuracies.append(train_accuracy)
            testing_accuracies.append(test_accuracy)
            print(
                f"Epoch {epoch}, Loss: {loss}, Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")

        # Plotting
    plt.figure(figsize=(12, 5))

    # Plot training and testing accuracies
    plt.subplot(1, 2, 1)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(testing_accuracies, label='Testing Accuracy')
    plt.title('Training and Testing Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
