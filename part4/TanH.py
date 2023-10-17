from Layers import Base
import numpy as np

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.activation_tensor = np.tanh(input_tensor)
        return self.activation_tensor

    def backward(self, error_tensor):
        next_e_tensor = error_tensor * (1 - self.activation_tensor**2)
        return next_e_tensor