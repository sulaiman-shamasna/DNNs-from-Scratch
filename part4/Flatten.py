from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_dim = input_tensor.shape
        #@TODO changed for Ex3
        if len(input_tensor.shape) == 4:
            b, self.c, self.m, self.n = input_tensor.shape
            reshaped_tensor = input_tensor.reshape(b,self.c * self.m * self.n)
        else:
            reshaped_tensor = input_tensor
        return reshaped_tensor

    def backward(self,error_tensor):
        if len(self.input_dim)==4:
            b, x = error_tensor.shape
            reshaped_tensor = error_tensor.reshape(b,self.c, self.m, self.n)
        else:
            reshaped_tensor = error_tensor
        return reshaped_tensor

