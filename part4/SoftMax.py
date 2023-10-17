from Layers import Base
import numpy as np

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        #calculate the SoftMax function - there are no weights
        #page 14, eq 13
        #X = batch size x (input size+1) and we get the max of entire batch size
        temp_tensor = np.exp(input_tensor - np.max(input_tensor)) #element wise subtraction  of the max
        temp_sum_arr = np.sum(temp_tensor, axis=1).reshape(1, input_tensor.shape[0]) #sum of the exp func along the columns per sample vector
        self.pred_tensor = ((temp_tensor.T / temp_sum_arr)).T #element wise divide, but it only works matrix/row hence transpose
        #this is not the normal input X , but the output of a layer. we want to boost only one nueron in each layer for one input
        #of the batch, hence we sum along the columns and divide column wise.
        #for eg X = 9x4 and pred, Y = 9x4  / 9x1 element wise so for each column, one of the 4 nuerons is boosted

        return self.pred_tensor

    def backward(self, error_tensor):
        #The next layer error tensor(in backward direction) is simply the error tensor where input tensor is +ve
        #multiply the error tensor with the predicted tensor y
        #reshaped to column vector so that it can be subtracted element wise
        #page 16, eqn 14
        #En is batch size x (input size+1) , pred tensor = batch size x  (input_size +1),
        #this is the softmax layer end of the Network
        #we sum along the columns as that is N, page 16 along all the nuerons of the layer for each input of batch B
        #for eg 9x4 - (9x4 @ 9x4 => summed to 9x1 )
        temp_err_tensor = error_tensor - (np.sum(np.multiply(error_tensor, self.pred_tensor), axis=1).reshape(self.pred_tensor.shape[0], 1))
        # element wise multiply the predicted tensor with the error corrected tensor, to update the weights essentially
        bnxt_error_tensor = np.multiply(self.pred_tensor, temp_err_tensor)
        return bnxt_error_tensor
