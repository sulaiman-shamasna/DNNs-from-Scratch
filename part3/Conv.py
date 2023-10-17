import copy
from Layers import Base
import numpy as np
from scipy import signal

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.__optimizer = None
        self.__bias_optimizer = None
        #size is set in the backward pass, but always to zero ?
        self.g_weights = []
        self.g_bias = []
        #used in the property
        self.fan_in = np.prod(convolution_shape)
        self.fan_out = np.prod(convolution_shape[1:])*num_kernels
        self.weights = np.random.uniform(0, 1, size=(num_kernels, *convolution_shape))
        #making bias a 1xk vector throws unit test error
        self.bias = np.random.uniform(0, 1, size=(1, num_kernels))

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

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize([1, self.num_kernels],self.fan_in ,self.fan_out )
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        return

    def forward(self,input_tensor):

        self.input_tensor = copy.deepcopy(input_tensor)
        #extract the input dimensions later check 1 or 2D processing
        input_dim = input_tensor.shape
        b_size = input_dim[0]
        self.c_size = input_dim[1]


        if len(input_dim) == 3:
            #c channel , 1D
            h_w = int(self.convolution_shape[1]/2)
            v_w = 0
            padded_input_tensor = np.pad(self.input_tensor, ((0,0),(0,0),(h_w, h_w)), 'constant', constant_values=(0,0))
            h1_w = (self.convolution_shape[1] + 1) % 2;
            padded_input_tensor = np.pad(padded_input_tensor, ((0,0),(0,0),(h1_w,0)), 'constant', constant_values=(0,0))
            #padded_input_tensor = np.pad(padded_input_tensor, ((0,0),(0,0),(check_r[0],0)), 'constant', constant_values=(0,0))

        elif len(input_dim) ==4:
            h_w = int(self.convolution_shape[1]/2)
            v_w = int(self.convolution_shape[2]/2)
            #2D padding
            padded_input_tensor = np.pad(input_tensor, ((0,0),(0,0),(h_w, h_w),(v_w, v_w)), 'constant', constant_values=(0,0))
            #assymetric padding
            h1_w = (self.convolution_shape[1] + 1) % 2; v1_w = (self.convolution_shape[2] + 1) % 2
            padded_input_tensor = np.pad(padded_input_tensor, ((0,0),(0,0),(h1_w,0),(v1_w,0)), 'constant', constant_values=(0,0))


        # we have b, c, x, y
        # we want b, k, x, y
        #This is the right shape for the next input tesor b*num_k*x*y
        next_fwd_tensor = np.zeros((b_size, self.num_kernels, *input_tensor.shape[2:]))

        for b_ind in range(b_size):
            for k in range(self.num_kernels):
                kernel = self.weights[k,:]

                if (len(input_dim) == 4):
                    #this is in 3D shape, channel size x image size
                    corr = signal.correlate(padded_input_tensor[b_ind].reshape(self.c_size,*padded_input_tensor.shape[2:]), kernel, mode='same')

                    #its not doing the summation, it outputs N channel output
                    if (self.c_size>1):
                        sel_c = self.c_size//2
                        corr1 = corr[sel_c]
                    else:
                        corr1 = corr[0]

                    next_fwd_tensor[b_ind, k] = corr1[h_w:h_w + input_dim[2], v_w:v_w + input_dim[3]]

                if (len(input_dim) == 3):
                    corr = signal.correlate(padded_input_tensor[b_ind].reshape(self.c_size, *padded_input_tensor.shape[2:]),kernel, mode='same')
                    if (self.c_size>1):
                        sel_c = self.c_size//2
                        corr1 = corr[sel_c]
                    else:
                        corr1 = corr[0]
                    next_fwd_tensor[b_ind, k] = corr1[h_w:h_w+input_dim[2]]

                #add the same bias term for all elements convolved with a kernel. bias/kernel
                if self.num_kernels==1:
                    next_fwd_tensor[b_ind,k,:] = next_fwd_tensor[b_ind,k,:] + self.bias
                else:
                    next_fwd_tensor[b_ind,k,:] = np.add(next_fwd_tensor[b_ind,k,:], self.bias[0,k])

        #only take the downsampled output
        if (len(input_dim)==4):
            ds_next_fwd_tensor = next_fwd_tensor[:,:,::self.stride_shape[0], 0::self.stride_shape[1]]
        elif (len(input_dim)==3):
            ds_next_fwd_tensor = next_fwd_tensor[:,:,::self.stride_shape[0]]

        return ds_next_fwd_tensor

    def backward(self,error_tensor):
        error_dim = error_tensor.shape
        b_size= error_dim[0]

        self.g_weights = np.zeros((self.num_kernels, *self.convolution_shape))
        self.g_bias = np.zeros((1, self.num_kernels))
        next_bwd_tensor = np.zeros((b_size, self.c_size, *self.input_tensor.shape[2:]))

        if len(error_dim) == 3:
            #c channels x 1D
            h_w = int(self.convolution_shape[1]/2)
            #create a larger matrix and assign accordingly
            padded_error_tensor = np.zeros((*error_dim[0:2],*self.input_tensor.shape[2:] ))
            #assign the error tensor to the strided positions
            padded_error_tensor[:,:,::self.stride_shape[0]] = error_tensor
            h1_w = (self.convolution_shape[1]+1)%2
            #anytime u do padding, do uneven padding for even kernels
            padded_error_tensor = np.pad(padded_error_tensor, ((0,0),(0,0),(h1_w, 0)), 'constant', constant_values=(0,0))
            #padding the input tensor as per the task description half kernel height
            padded_input_tensor = np.pad(self.input_tensor, ((0,0),(0,0),(h_w, h_w)), 'constant', constant_values=(0,0))

        elif len(error_dim) ==4:
            h_w = int(self.convolution_shape[1]/2)
            v_w = int(self.convolution_shape[2]/2)
            #2D padding
            padded_error_tensor = np.zeros((*error_dim[0:2],*self.input_tensor.shape[2:] ))
            #assign the padded tensor to the strided positions
            padded_error_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]] = error_tensor
            #pad only one side if the kernel is even
            h1_w = (self.convolution_shape[1]+1)%2 ; v1_w = (self.convolution_shape[2]+1)%2
            padded_error_tensor = np.pad(padded_error_tensor, ((0,0),(0,0),(h1_w, 0),(v1_w, 0)), 'constant', constant_values=(0,0))
            #as given in task description pad with half the kernel width and height
            padded_input_tensor = np.pad(self.input_tensor, ((0,0),(0,0),(h_w, h_w),(v_w, v_w)), 'constant', constant_values=(0,0))

        #error_tensor shape is b_size x num_kernels x image size
        #output is b_size, channel_size x image size
        for b_ind in range(b_size):
            for c_ind in range(self.c_size):
                if len(error_dim) ==3:
                    kernel = self.weights[:, c_ind, :]
                    kernel = kernel[::-1] #flipping along the kernel axis
                    corr = signal.convolve(padded_error_tensor[b_ind],kernel, mode='same')
                    if (self.num_kernels>1):
                        corr1 = corr[1]
                    else:
                        corr1 = corr[0]
                    next_bwd_tensor[b_ind, c_ind] = corr1[0:self.input_tensor.shape[2]]

                elif len(error_dim)==4:
                    kernel = self.weights[:,c_ind,:,:]
                    kernel = kernel[::-1] #flipping along the kernel axis
                    corr = signal.convolve(padded_error_tensor[b_ind],kernel,mode='same')
                    if (self.num_kernels>1):
                        corr1 = corr[1]
                    else:
                        corr1 = corr[0]
                    #doesnt work if you remove the extra line padding
                    next_bwd_tensor[b_ind, c_ind] = corr1[:self.input_tensor.shape[2],:self.input_tensor.shape[3]]

        #input image = batch size x channels x image size
        #g_weights is size num_kernels x num_channels x kernel size
        #error tensor = batch size x num of kernels x image size
        #Calculating the gradient and update to the bias, used valid mode
        for k_ind in range(self.num_kernels):
            for c_ind in range(self.c_size):
                if (len(error_dim)==3):
                    #to generate the g_weights in 1D
                    #for the batch size, per channel, img size
                    #all the batches , one channel
                    #@TODO check this we take only first element of the batch?
                    in_tensor = padded_input_tensor[:, c_ind].reshape(b_size, *padded_input_tensor.shape[2:])
                    e_kernel = padded_error_tensor[:, k_ind].reshape(b_size, *padded_error_tensor.shape[2:])
                    corr = signal.correlate(in_tensor, e_kernel, mode='valid')
                    self.g_weights[k_ind, c_ind] = corr[0]

                if (len(error_dim)==4):
                    in_tensor = padded_input_tensor[:, c_ind].reshape(b_size, *padded_input_tensor.shape[2:])
                    e_kernel = padded_error_tensor[:, k_ind].reshape(b_size, *padded_error_tensor.shape[2:])
                    corr = signal.correlate(e_kernel, in_tensor, mode='valid')
                    self.g_weights[k_ind, c_ind] = np.rot90(np.rot90(corr[0]))

            self.g_bias[0,k_ind] = np.sum(error_tensor[:,k_ind,:])

        if self.__optimizer:
            self.weights = self.__optimizer.calculate_update(self.weights, self.g_weights)
            self.bias = self.__bias_optimizer.calculate_update(self.bias, self.g_bias)

        return next_bwd_tensor
