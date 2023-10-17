from Layers import Base
import numpy as np

class Dropout(Base.BaseLayer):
    def __init__(self, probabibility):
        super().__init__()
        self.keep_p = probabibility

    def forward(self, input_tensor):
        self.mask = np.random.binomial(1, self.keep_p, size=input_tensor.shape) / self.keep_p
        if (self.testing_phase==False):
            output_tensor = input_tensor * self.mask
        else:
            output_tensor = input_tensor
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = error_tensor * self.mask
        return  output_tensor
