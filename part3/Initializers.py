import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        initialized_weights = np.array([self.value]*np.ones(weights_shape))
        return initialized_weights

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        initialized_weights = np.random.uniform(0,1, size=weights_shape)
        return initialized_weights

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        sigma = np.sqrt(2/(fan_in+fan_out))
        initialized_weights = np.random.normal(0, sigma, size=weights_shape)
        return initialized_weights


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out
        sigma = np.sqrt(2/fan_in)
        initialized_weights =np.random.normal(0, sigma, weights_shape)

        return initialized_weights
