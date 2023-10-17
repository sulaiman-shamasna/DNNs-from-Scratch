from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        b, self.c, self.m, self.n = input_tensor.shape
        reshaped_tensor = input_tensor.reshape(b,self.c * self.m * self.n)
        return reshaped_tensor

    def backward(self,error_tensor):
        b, x = error_tensor.shape
        reshaped_tensor = error_tensor.reshape(b,self.c, self.m, self.n)
        return reshaped_tensor

