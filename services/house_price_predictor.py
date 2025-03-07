import numpy as np

class HousePricePredictor:
    def __init__(self):
        pass

    #univariate linear regression predictor
    def predict(self, w, b, x):
        #amount of data points
        m = x.shape[0]
        predictions = np.zeros(m)

        for i in range(m):
            predictions[i] = w * x[i] + b

        return predictions
