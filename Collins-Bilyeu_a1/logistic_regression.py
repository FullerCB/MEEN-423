import numpy as np


def fit(X_data, y_data, eta, n_epochs):

    X = np.concatenate([np.ones((X_data.shape[0], 1)), np.array(X_data)], axis=1)
    y = np.array(y_data)

    Ndata, Nfeat = X_data.shape
    weights = np.zeros(Nfeat+1)
    accuracy = np.zeros(n_epochs*Ndata)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for epoch in range(n_epochs):
        # weight loop
        for sample in range(Ndata):

            prediction = sigmoid(X @ weights)
            gradient = (1 / Ndata) * (X.T @ (prediction - y))
            weights -= eta * gradient


    return weights

def predict(x, weights):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    x = np.insert(x, 0, 1)
    prediction = sigmoid(x @ weights)
    if prediction >= 0.5:
        return 1
    else:
        return 0