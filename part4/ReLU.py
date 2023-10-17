from Layers import Base
import numpy as np

class ReLU(Base.BaseLayer):
    #inheriting the base layer
    def __init__(self):
        #super constructor to inherit the trainable variable
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #calculate the ReLU function - there are no weights
        #page 12, no weights as this is only activation layer on top of the fully connected layer
        fnxt_input_tensor = np.maximum(self.input_tensor, 0)
        return fnxt_input_tensor

    def backward(self, error_tensor):
        #The next layer error tensor(in backward direction) is simply the error tensor where input tensor is +ve
        #due to chain rule of derivatives, the partial derivative wrt x = 1 , so only e term is left - page 13
        bnxt_error_tensor = error_tensor
        bnxt_error_tensor[self.input_tensor<=0] = 0
        return bnxt_error_tensor