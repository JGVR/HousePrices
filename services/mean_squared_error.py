class Mse:
    def __init__(self):
        pass

    # > calculate mean squared error
    # > formula: 1/2m * sum(prediction[i] - y[i])^2
    def call(self, predictions, x, y):
        m = x.shape[0]
        sum_of_predictions = 0

        for i in range(m):
            sum_of_predictions += (predictions[i] - y[i]) ** 2

        return 1/(2*m) * sum_of_predictions