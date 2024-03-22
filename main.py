import pandas as pd
from preprocessing import preprocessing
from training import automatic_tuning, manual_tuning


def main():
    train_data, test_data = preprocessing()

    x_train = train_data.drop('Species', axis=1).values
    y_train = pd.get_dummies(train_data['Species']).values
    x_test = test_data.drop('Species', axis=1).values
    y_test = pd.get_dummies(test_data['Species']).values

    choice = input("Do you want to perform manual tuning or automatic tuning? (manual/auto): ").lower()
    if choice == 'manual':
        nn, losses, train_accuracy, test_accuracy = manual_tuning(x_train, y_train, x_test, y_test)
        nn.evaluate(x_test, y_test)
    elif choice == 'auto':
        nn, losses, train_accuracy, test_accuracy = automatic_tuning(x_train, y_train, x_test, y_test)
        nn.evaluate(x_test, y_test)
    else:
        print("Invalid input. Please enter 'manual' or 'auto'.")


if __name__ == "__main__":
    main()
