from Layers import Base
import numpy as np

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.activation_tensor = 1 /(1 + np.exp(-1 * input_tensor))
        return self.activation_tensor

    def backward(self, error_tensor):
        next_e_tensor = error_tensor * self.activation_tensor * (1-self.activation_tensor)
        return next_e_tensor