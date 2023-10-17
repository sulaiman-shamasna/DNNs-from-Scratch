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
        #@TODO replace with np.pad
        t_bwd_tensor = np.concatenate((t_bwd_tensor, np.zeros((*error_tensor.shape,
                                                    self.pooling_shape[0]*self.pooling_shape[1]-1))),axis=4 )
        t_bwd_tensor[:] = 0 #fill with zeros
        op_shape = t_bwd_tensor.shape
        #reshape to nx4 where the poolinng shape becomes columns
        t_bwd_tensor = t_bwd_tensor.reshape(-1, self.pooling_shape[0]*self.pooling_shape[1])
        e_tensor = error_tensor.ravel().reshape(-1,1)
        ind = np.arange(e_tensor.shape[0])[:, np.newaxis]
        t_bwd_tensor[ind, self.max_ind[:,None]] = e_tensor

        next_bwd_tensor = np.zeros_like(self.input_tensor)
        t_bwd_tensor = t_bwd_tensor.reshape(op_shape);
        s1 = self.stride_shape[0]; s2 = self.stride_shape[1]
        p1 = self.pooling_shape[0]; p2 = self.pooling_shape[1]
        c = self.input_tensor.shape[1]; b = self.input_tensor.shape[0]
        #the m, n for loops are looping over the windows
        for m in range(self.output_shape[0]): #store output shape and resue here
            #this is to take care of overlapping pooling windows, else does not matter
            #the number of windows that are created for the pooling
            for n in range(self.output_shape[1]): #store output shape and reuse here
                #this is to take care of overlapping pooling windows, else it does not matter
                next_bwd_tensor[:,:,m*s1:m*s1+p1,n*s2:n*s2+p2] = \
                          next_bwd_tensor[:,:,m*s1:m*s1+p1,n*s2:n*s2+p2] \
                          + t_bwd_tensor[:, :, m, n,:].reshape(b, c, p1, p2) #pooling shape 2x2
                          #t_bwd_tensor is of shape n x 4 and we reshape it and add up its contribution where computed
                          #do all the channels at one time

        return next_bwd_tensor
