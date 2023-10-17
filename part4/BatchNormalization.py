from Layers import Base
from Layers import Helpers
import numpy as np
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
#        self.initialize()
        self.g_weights = []
        self.g_bias = []
        self.__optimizer = None
        self.mu = []
        self.sigma = []
        self.first = True
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)
        #self.weights = np.random.uniform(0,1, size=(self.channels))

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize([1, self.channels],self.channels ,self.channels )
        self.weights = weights_initializer.initialize(self.channels, self.channels, self.channels)
        return

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.__optimizer = opt
        self.__bias_optimizer = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self.g_weights

    @gradient_weights.setter
    def gradient_weights(self, g_tensor):
        self.g_weights = g_tensor

    @property
    def gradient_bias(self):
        return self.g_bias

    @gradient_bias.setter
    def gradient_bias(self, bias_vector):
        self.g_bias = bias_vector

    def calculate_regularization_loss(self):
        return self.__optimizer.regularizer.norm(self.weights)

    def reformat(self, tensor):
        if len(tensor.shape)==4:
            t_dim = tensor.shape
            r_tensor = tensor.reshape(t_dim[0],t_dim[1],t_dim[2]*t_dim[3])
            r_tensor = np.transpose(r_tensor, [0, 2, 1])
            r_tensor = r_tensor.reshape(t_dim[0]*t_dim[2]*t_dim[3], t_dim[1])
        else:
            t_dim = self.t_shape
            r_tensor = tensor.reshape(t_dim[0], t_dim[2]*t_dim[3], t_dim[1])
            r_tensor = np.transpose(r_tensor, [0, 2, 1])
            r_tensor = r_tensor.reshape(self.t_shape)

        return r_tensor

    def forward(self, input_tensor):
        #there are trainable parameters - exam question
        self.t_shape = input_tensor.shape
        self.input_tensor = copy.deepcopy(input_tensor)
        if len(input_tensor.shape)==4:
            i_tensor = self.reformat(input_tensor)
        else:
            i_tensor = input_tensor

        if self.first == True:
            self.mu = np.mean(i_tensor, axis=0).reshape(1, -1)
            self.sigma = np.var(i_tensor, axis=0).reshape(1, -1)
            self.first = False

        if (self.testing_phase==False):
            #only for training part
            #testing phase has different requirements of the moving avg
            self.mu_b = np.mean(i_tensor, axis=0).reshape(1, -1)
            self.sigma_b = np.var(i_tensor, axis=0).reshape(1, -1)
            self.x_tensor = (i_tensor - self.mu_b) / (np.sqrt(self.sigma_b + np.finfo(float).eps))
            o_tensor = self.x_tensor * self.weights  + self.bias
            #calculate the moving average on the training set
            self.mu = 0.8 * self.mu +  0.2 * self.mu_b
            self.sigma = 0.8 * self.sigma + 0.2 * self.sigma_b
        else:
            self.x_tensor = (i_tensor - self.mu) / (np.sqrt(self.sigma + np.finfo(float).eps))
            o_tensor = self.x_tensor * self.weights  + self.bias

        #print(o_tensor.shape,'fwd pass o_tensor for 4D')
        if len(input_tensor.shape)==4:
            output_tensor = self.reformat(o_tensor)
        else:
            output_tensor = o_tensor

        return output_tensor

    def backward(self,  error_tensor):
        #might also smoothen the loss surface
        #generally improves performance in networks

        #handle the convolution case
        if (len(self.t_shape)==4):
            i_tensor = self.reformat(self.input_tensor)
            e_tensor = self.reformat(error_tensor.reshape(self.t_shape))
        else:
            i_tensor = self.input_tensor
            e_tensor = error_tensor

        next_bwd_tensor = Helpers.compute_bn_gradients(e_tensor, i_tensor, self.weights, self.mu_b, self.sigma_b)

        self.g_weights = np.sum(e_tensor*self.x_tensor, axis=0)
        self.g_bias = np.sum(e_tensor, axis=0)

        if len(self.input_tensor.shape)==4:
            n_bwd_tensor = self.reformat(next_bwd_tensor)
        else:
            n_bwd_tensor = next_bwd_tensor

        if self.__optimizer:
            self.weights = self.__optimizer.calculate_update(self.weights, self.g_weights)
            self.bias = self.__bias_optimizer.calculate_update(self.bias, self.g_bias)

        return n_bwd_tensor