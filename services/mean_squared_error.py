import numpy as np

class Mse:
    def __init__(self):
        pass

    # > calculate mean squared error
    # > formula: 1/2m * sum(prediction - y[i])^2
    def call(self, w, b, X, y):
        m,n = X.shape
        sum_of_predictions = 0

        for i in range(m):
            prediction = np.dot(X[i], w) + b
            for j in range(n):
                sum_of_predictions += (prediction - y[j]) ** 2

        return 1/(2*m) * sum_of_predictions