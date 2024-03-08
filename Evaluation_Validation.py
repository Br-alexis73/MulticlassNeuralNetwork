import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(y_true, y_pred):
    # Assuming y_pred is probabilities, we take the argmax to get predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)  # This will convert to (623,)

    # Now y_true and y_pred_classes should have the same shape
    return np.mean(y_true == y_pred_classes)

def plot_learning_curves(losses):
    plt.plot(losses)
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
