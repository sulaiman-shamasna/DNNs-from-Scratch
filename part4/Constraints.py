import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        g_weights = weights * self.alpha
        return g_weights

    def norm(self, weights):
        n_weights = np.sum(weights**2) * self.alpha
        return n_weights



class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        g_weights = self.alpha * np.abs(weights)/weights
        return g_weights

    def norm(self, weights):
        n_weights = np.sum(np.abs(weights)) * self.alpha
        return n_weights
