import numpy as np
from data_prep import features, targets, features_test, targets_test

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
# weights should be small such that the input to the sigmoid is in the linear region near 0 (not squashed at the high and low ends)
# inititialize randomly so that they all have different starting values, and diverge, breaking symmetry
# a good value for the scale is 1/sqrt(n) - where n in the number of input units
# this keeps the sigmoid low for increasing numbers of input units
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Calculate the output
        output = sigmoid(np.dot(x, weights))

        # Calculate the error
        error = y - output

        # Calculate change in weights
        # new weight = old weight + error
        del_w += error * output * (1 - output) * x

        # Update weights
        # weights = weight + (eta * del_w)/m
    weights += (del_w * learnrate)/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
