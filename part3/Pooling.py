from Layers import Base
import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        super(Pooling, self).__init__()
        self.trainable = False


    def forward(self, input_tensor):
        #only 2D case
        self.input_tensor = input_tensor
        b, c, row , col = input_tensor.shape
        srow , scol = self.stride_shape
        prow , pcol = self.pooling_shape
        self.output_shape = ((row - prow) // srow + 1,
                        (col - pcol) // scol + 1)

        A_w = as_strided(input_tensor, shape=(input_tensor.shape[0:2]+self.output_shape+self.pooling_shape),
                         strides=(input_tensor.strides[:2]+(srow*input_tensor.strides[2], scol*input_tensor.strides[3])+
                                  input_tensor.strides[2:]))
        # Return the result of pooling
        next_fwd_tensor = A_w.max(axis=(4,5))
        r_A_w = A_w.reshape(*A_w.shape[:-2],-1)
        self.max_ind = np.argmax(r_A_w, axis=4).ravel()
        return next_fwd_tensor

    def backward(self, error_tensor):

        #since we have stored indices in flattened array anyways , axis = reshape later
        t_bwd_tensor = np.expand_dims(error_tensor, axis=(4))
        #create the array that has the pooling shape x pooling shape extra columns or new dimension...
        t_bwd_tensor = np.concatenate((t_bwd_tensor, np.zeros((*error_tensor.shape,
                                                    self.pooling_shape[0]*self.pooling_shape[1]-1))),axis=4 )
        t_bwd_tensor[:] = 0 #fill with zeros
        op_shape = t_bwd_tensor.shape
        #reshape to nx4 where the poolinng shape becomes columns
        a = t_bwd_tensor.reshape(-1, self.pooling_shape[0]*self.pooling_shape[1])
        e = error_tensor.ravel()
        #@TODO fix this for loop
        for i in range(a.shape[0]):
            a[i, self.max_ind[i]] = e[i]

        #Reshape back to expected shape
        #next_bwd_tensor = copy.deepcopy(a.reshape(op_shape))
        next_bwd_tensor = np.zeros_like(self.input_tensor)
        #next_bwd_tensor = np.pad(next_bwd_tensor, ((0, 0), (0, 0), (0, self.h_w), (0, self.v_w)), 'edge')

        a = a.reshape(op_shape);
        #print(self.stride_shape, self.pooling_shape)
        #i and j take care of the stride overlaps
        #the m, n for loops are looping over the windows
        #the b and c for loops are looping over the batch and channel sizes
        for b in range(self.input_tensor.shape[0]):
            for c in range(self.input_tensor.shape[1]):
                j=-1*self.stride_shape[0]
                #for each pooling window, for eg 2x4 windows created check each window
                for m in range(self.output_shape[0]): #store output shape and resue here
                    #this is to take care of overlapping pooling windows, else does not matter
                    j = j + self.stride_shape[0]
                    i=-1*self.stride_shape[1]
                    #the number of windows that are created for the pooling
                    for n in range(self.output_shape[1]): #store output shape and reuse here
                        #this is to take care of overlapping pooling windows, else it does not matter
                        i = i + self.stride_shape[1]
                        next_bwd_tensor[b,c,j:j+self.pooling_shape[0],i:i+self.pooling_shape[1]] = \
                            next_bwd_tensor[b,c,j:j+self.pooling_shape[0],i:i+self.pooling_shape[1]] \
                            + a[b, c, m, n,:].reshape(self.pooling_shape[0], self.pooling_shape[1]) #pooling shape 2x2
                        #a is of shape n x 4 and we reshape it and add up its contribution where computed

        return next_bwd_tensor
