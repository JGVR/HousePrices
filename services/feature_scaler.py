import numpy as np

class FeatureScaler:
    # > simple feature scaling
    @staticmethod
    def simple_feature_scaling(data):
        m = data.shape[0]
        scaled_data = np.zeros(m)
        for i in range(m):
            scaled_data[i] = data[i] / data.max()

        return scaled_data