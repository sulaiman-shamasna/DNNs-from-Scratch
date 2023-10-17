from Optimization import Optimizers
from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    #inheriting the base layer and include the super constructor
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        #set the trainable to True from the Baselayer class
        self.trainable = True
        #private variable optimizer associated with the property, which later links to Optimizer.sgd(learning rate)
        self.__optimizer = None
        self.gradient_tensor = None
        #initialize the weights randomly between 0 - 1, ensure the size is input_size+1
        #this is W' in page 9, eqn 9
        self.weights = np.random.uniform(0, 1, size=(self.input_size+1, output_size))
        self.fan_in = self.input_size
        self.fan_out = self.output_size

    #this is the get property, there are 2 ways to implement get_ , set_
    #do not mix the two ways.
    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.__optimizer = opt

    #this is the get property
    @property
    def gradient_weights(self):
        return self.gradient_tensor

    #this is the set property
    #can also use get_gradient_weights and set_gradient_weights, but not together with setter
    #somehow that did not work
    @gradient_weights.setter
    def gradient_weights(self, g_tensor):
        self.gradient_tensor = g_tensor

    def initialize(self, weights_initializer, bias_initializer):
        #@TODO check this, it works in multiple cases
        self.weights[-1,:] = bias_initializer.initialize([1, self.output_size],self.fan_in, self.fan_out)
        self.weights[:-1,:] = weights_initializer.initialize([self.input_size, self.output_size], self.fan_in, self.fan_out)
        return

    def forward(self, input_tensor):
        #concateneate the bias related term 1 directly along columns, same as np.concatenate,
        #but do not need to specify axis with this. similar to np.r_
        self.input_tensor =np.c_[input_tensor, np.ones(input_tensor.shape[0])]
        #calulate the inputs to the next fwd layer
        #page 9, eqn 9 =>  X'@ W'. Assume input tensor is already in the transposed shape
        #X' is batch size x (input size+1) eg. X' = 9x4
        fnxt_input_tensor = self.input_tensor @ self.weights
        return fnxt_input_tensor

    def backward(self, error_tensor):
        #page 10, eqn 10 -> En' @ W'.T
        #En = batch size x (output size) and W'.T = output size x (input size+1)
        # En = 9x output and W'.T = output x 4
        temp_error_tensor = error_tensor @ self.weights.T
        #calculate the error tensor to be sent to the previous layer (next in backpropogation)
        #so ignore the last column which is the update to the bias terms
        #do not include bias terms in the back prop of weights as partial deriv wrt to weights sets bias term = 0
        #En is the same shape as the output Y'
        bnxt_error_tensor = temp_error_tensor[:,:-1]
        #calc the gradients to update the weights wrt X
        #X' is batch size x (input_size+1) , En = batch size x output size
        #X'.T @ En ==> eqn 11, page 10, use the bias terms as well
        self.gradient_tensor = self.input_tensor.T @ error_tensor
        if self.__optimizer:
            #update the weights only if the optimizer is set when we call Optimizer.Sgd()
            self.weights = self.__optimizer.calculate_update(self.weights, self.gradient_tensor)
        return bnxt_error_tensor


