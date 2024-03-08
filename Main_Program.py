import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from NN import MultiClassNeuralNetwork
from Evaluation_Validation import calculate_accuracy, plot_learning_curves
from Evaluation_Cross_Validation import cross_validation
from Evaluation_Curves import hyperparameter_tuning, multi_layer_hyperparameter_tuning


# def get_user_input():
#     # Ask the user to choose between single-layer or multi-layer network
#     network_type = input("Choose network type (Single or Multi-layer): ").strip().lower()
#
#     if network_type == "single":
#         # If single layer, only ask for the activation function
#         activation_function = input(
#             "Enter the activation function for the single layer (e.g., sigmoid, relu): ").strip().lower()
#         learning_rate = float(input("Enter the learning rate (e.g., 0.01): ").strip())
#         return {
#             "network_type": "single",
#             "activation_function": activation_function,
#             "learning_rate": learning_rate
#         }
#
#     elif network_type == "multi":
#         # If multi-layer, ask for number of neurons, layers, and activation functions
#         num_neurons = input("Enter the number of neurons in each hidden layer separated by comma (e.g., 64,64): ")
#         num_layers = int(input("Enter the number of layers in the network: ").strip())
#         inner_activation_function = input(
#             "Enter the inner layers activation function (e.g., sigmoid, relu): ").strip().lower()
#         output_activation_function = input(
#             "Enter the output activation function (e.g., softmax, sigmoid): ").strip().lower()
#         loss_function = input("Enter the loss function (e.g., cross_entropy): ").strip().lower()
#         learning_rate = float(input("Enter the learning rate for the multi-layer network (e.g., 0.01): ").strip())
#         return {
#             "network_type": "multi",
#             "num_neurons": [int(n) for n in num_neurons.split(',')],
#             "num_layers": num_layers,
#             "inner_activation_function": inner_activation_function,
#             "output_activation_function": output_activation_function,
#             "loss_function": loss_function,
#             "learning_rate": learning_rate
#         }
#
#     else:
#         print("Invalid network type selected.")
#         return None


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
    hidden_layers = [5]  # Example: One hidden layer with 5 neurons
    nn = MultiClassNeuralNetwork(input_size, hidden_layers, output_size)

    # Prepare training data
    x_train = train_data.drop('Species', axis=1).values
    y_train = train_data['Species'].values

    # Prepare testing data
    x_test = test_data.drop('Species', axis=1).values
    y_test = test_data['Species'].values

    epochs = 100  # Number of times to loop through the entire dataset
    learning_rate = 0.01  # Learning rate for the optimizer

    for epoch in range(epochs):
        # Forward propagation
        predictions = nn.feedforward(x_train)

        # Calculate loss
        loss = nn.cross_entropy_loss(y_train, predictions)

        # Backpropagation
        gradients = nn.backpropagation(y_train, predictions)

        # Update weights
        nn.update_weights(gradients, learning_rate)

        if epoch % 10 == 0:  # Print the loss every 10 epochs
            print(f"Epoch {epoch}, Loss: {loss}")

    # Evaluate the network's performance on training data
    train_predictions = nn.feedforward(x_train)
    train_accuracy = calculate_accuracy(y_train, train_predictions)
    print(f"Training accuracy: {train_accuracy}")

    # Evaluate the network's performance on testing data
    test_predictions = nn.feedforward(x_test)
    test_accuracy = calculate_accuracy(y_test, test_predictions)
    print(f"Testing accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()