import copy
from Optimization import *
from Layers import *

class NeuralNetwork:
    def __init__(self, optimizer):
        #set the optimizer thats passed here
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None


    def forward(self):
        #data layer has d dimensional data
        #returns a set of input tensor and label tensor
        self.input_tensor, self.label_tensor = self.data_layer.next()
        #pass the data through all the layers
        #unsure if the input tensor is allowed to change, so make copy to be safe
        activation_tensor = self.input_tensor.copy()
        for layer in self.layers:
            activation_tensor = layer.forward(activation_tensor)

        #initialized in the given test functions, so we only call it here
        #calculate the loss which should be returned
        calc_loss = self.loss_layer.forward(activation_tensor,self.label_tensor)
        #perhaps not needed , creating a copy of the loss
        copy_calc_loss = calc_loss.copy()
        return copy_calc_loss

    def backward(self, label_tensor):
        #the layers are initialized in the given test functions
        #@start from the loss layer
        error_tensor = self.loss_layer.backward(label_tensor)
        #propogate the error tensor through the network
        #use the reversed way to go backwards in the layers
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            #make a deep copy of the optimizer thats been passed to the class
            #so that the layer optimizer is not linked to the optimizer that was originally set
            #so that future changes in self.optimizer will not impact this layer.optimizer here
            #since the test func sets the optimizer for each layer before appending to the network
            #use the optimizer property to set
            copy_optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = copy_optimizer
        #layer is appended to the list of layers
        #append is defined in python to any list
        self.layers.append(layer)
        return

    def train(self, iterations):
        for idx in range(iterations):
            #calling the forward func defined here, will pass through all layers and return the scalar loss.
            calc_loss = self.forward()
            #just do one backward pass for each iterations
            error = self.backward(self.label_tensor)
            #store the calculated loss to be plotted later
            self.loss.append(calc_loss)
        return

    def test(self,input_tensor):
        activation_tensor = input_tensor.copy()
        #one single forward pass through all the layers to get the final softmax output
        for layer in self.layers:
            activation_tensor = layer.forward(activation_tensor)

        #return the final output layer/result to calculate the accuracy
        return activation_tensor
