from Layers import Base
from Layers import TanH
from Layers import Sigmoid
from Layers import FullyConnected
import numpy as np
import copy

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.hidden_state = np.zeros(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.__memorize = False
        self.g_weights_xhh = None
        self.g_weights_hy = None
        #self.g_bias = None
        self.__optimizer = None
        self.__bias_optimizer = None
        self.first = True
        self.weights_xhh = np.random.uniform(0,1, size=(self.input_size+self.hidden_size+1, self.hidden_size))
        self.weights_hy = np.random.uniform(0,1 , size=(self.hidden_size+1, self.output_size))
        self.bias = np.ones((1,self.output_size))
        self.tanh_layer = TanH.TanH()
        self.sigmoid_layer = Sigmoid.Sigmoid()
        self.fc_layer1 = FullyConnected.FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.fc_layer2 = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.hidden_state_prev = []

    @property
    def memorize(self):
        return self.__memorize

    @memorize.setter
    def memorize(self, mem):
        self.__memorize = mem

    @property
    def gradient_weights(self):
        return self.g_weights_xhh

    @gradient_weights.setter
    def gradient_weights(self, g_tensor):
        self.g_weights_xhh = g_tensor

    @property
    def weights(self):
        return self.weights_xhh

    @weights.setter
    def weights(self, wts):
        self.weights_xhh = wts

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.__optimizer = opt

    def calculate_regularization_loss(self):
        return self.__optimizer.regularizer.norm(self.weights_xhh)

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_layer1.initialize(weights_initializer, bias_initializer)
        self.fc_layer2.initialize(weights_initializer, bias_initializer)
        self.weights_xhh = self.fc_layer1.weights
        self.weights_hy = self.fc_layer2.weights
        return

    def forward(self, input_tensor):
        #we have the 0th dim as time = batch dim
        self.input_tensor = input_tensor
        #fix the size and initialize
        self.fc_layer1.weights = self.weights_xhh
#        self.fc_layer1.gradient_weights = self.g_weights_xhh
        self.fc_layer2.weights = self.weights_hy
#        self.fc_layer2.gradient_weights = self.g_weights_hy
        next_fwd_tensor = np.zeros((input_tensor.shape[0], self.output_size))

        #hidden_state size is 1 x hidden size
        if self.__memorize==False:
            #set the (t-1)th hidden state to all zeros
            hidden_state_t_1 = np.zeros((1,self.hidden_size))
        else:
            #set the (t-1)th hidden state to previous one
            if self.first==False:
                hidden_state_t_1 = self.hidden_state_prev
            else:
                self.first=False
                hidden_state_t_1 = np.zeros((1, self.hidden_size))

        self.fc_layer1_input = np.zeros((input_tensor.shape[0], self.input_size+self.hidden_size))
        self.tanh_layer_activation = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.fc_layer2_input = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.sigmoid_layer_activation = np.zeros((input_tensor.shape[0], self.output_size))

        for i in range(input_tensor.shape[0]):
            if i==0:
                x_tilda = np.c_[input_tensor[i,None], hidden_state_t_1]
            else:
                x_tilda = np.c_[input_tensor[i, None], self.hidden_state]
            tanh_input = self.fc_layer1.forward(x_tilda)
            #saving the state
            self.fc_layer1_input[i] = x_tilda[None,:]
            self.hidden_state = self.tanh_layer.forward(tanh_input)
            self.tanh_layer_activation[i] = self.tanh_layer.activation_tensor
            sigmoid_input = self.fc_layer2.forward(self.hidden_state)
            self.fc_layer2_input[i] = self.hidden_state
            next_fwd_tensor[i] = self.sigmoid_layer.forward(sigmoid_input)
            self.sigmoid_layer_activation[i] = self.sigmoid_layer.activation_tensor

        #weights sharing across time domain , similar to weight sharing in the conv layer
        #keep the context information across time
        self.hidden_state_prev = self.hidden_state
        self.weights_xhh = self.fc_layer1.weights
        self.weights_hy = self.fc_layer2.weights

        return next_fwd_tensor

    def backward(self, error_tensor):
        self.bwd_hidden_state = np.zeros((1, self.hidden_size))
        self.g_weights_xhh = np.zeros_like(self.weights_xhh)
        self.g_weights_hy = np.zeros_like(self.weights_hy)
        next_bwd_tensor = np.zeros((error_tensor.shape[0], self.input_size))

        for i in range(error_tensor.shape[0])[::-1]:
            self.sigmoid_layer.activation_tensor = self.sigmoid_layer_activation[i]
            sigmoid_output = self.sigmoid_layer.backward(error_tensor[i])
            self.fc_layer2.input_tensor = np.c_[self.fc_layer2_input[i, None], 1]
            fc2_output = self.fc_layer2.backward(sigmoid_output[None,:])
            self.tanh_layer.activation_tensor = self.tanh_layer_activation[i]
            tanh_output = self.tanh_layer.backward(fc2_output+self.bwd_hidden_state)
            self.fc_layer1.input_tensor = np.c_[self.fc_layer1_input[i, None] , 1]
            fc1_output = self.fc_layer1.backward(tanh_output)
            #splitting the ouput of the fc layer 1 into two parts
            next_bwd_tensor[i] = fc1_output[:,:self.input_size]
            self.bwd_hidden_state = fc1_output[:,self.input_size:]
            #updating the gradients wrt weights
            self.g_weights_xhh = self.g_weights_xhh + self.fc_layer1.gradient_weights
            self.g_weights_hy = self.g_weights_hy + self.fc_layer2.gradient_weights

        if self.__optimizer:
            self.weights_xhh = self.__optimizer.calculate_update(self.weights_xhh, self.g_weights_xhh)
            self.weights_hy = self.__optimizer.calculate_update(self.weights_hy, self.g_weights_hy)

        return next_bwd_tensor
