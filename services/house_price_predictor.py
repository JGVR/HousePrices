import numpy as np

class HousePricePredictor:
    def __init__(self):
        pass

    def predict(self, w, b, x):
        return np.dot(x, w) + b
