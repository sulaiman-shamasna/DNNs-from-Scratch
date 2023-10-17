import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, pred_tensor, label_tensor):
        #pred tensor = y_hat, label tensor = y
        #pred tensor = batch size x input_size+1 , same as label tensor
        #page 17
        self.pred_tensor = pred_tensor
        #calculate the loss this is a scalar
        #add the small epsilon and eqn 15, page 17
        #only check those values which are non zero in the label tensor matrix, without this check we fail the unit test case
        #this works as python magic
        loss = -1*np.sum(np.log(self.pred_tensor[label_tensor!=0] + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        #prohibit divide by zero, by setting all zero values with a small offset
        #if the labels corresponding to those are zero,this will not change the output, if not, then we prevent /0
        self.pred_tensor[self.pred_tensor==0] = np.finfo(float).eps
        #element wise divide matrix by matrix
        #page 18, eq 16
        error_tensor = label_tensor/self.pred_tensor
        error_tensor = error_tensor*-1.
        return error_tensor