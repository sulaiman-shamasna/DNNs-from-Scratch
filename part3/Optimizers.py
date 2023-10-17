import numpy as np
class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor_next = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor_next

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.first = True

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.first ==True:
            self.v_k_1 =np.zeros_like(weight_tensor)
            self.first = False
        v_k = self.momentum_rate * self.v_k_1 - self.learning_rate*gradient_tensor
        weight_tensor_next = weight_tensor + v_k
        self.v_k_1 = v_k

        return weight_tensor_next

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.first = True
        #do not start with 0, as denom in c_v_k will become zero
        self.k = 1

    #combination of all the good optimizers and the best one we want to use
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.first ==True:
            self.v_k_1 =  np.zeros_like(weight_tensor)
            self.r_k_1 = np.zeros_like(weight_tensor)
            self.first = False

        v_k = self.v_k_1*self.mu +(1-self.mu)*gradient_tensor
        r_k = self.r_k_1*self.rho + (1-self.rho)* np.multiply(gradient_tensor,gradient_tensor)
        #bias correccted v_k and r_k
        c_v_k = v_k/(1-(self.mu**self.k))
        c_r_k = r_k/(1-(self.rho**self.k))
        weight_tensor_next = weight_tensor - self.learning_rate *(c_v_k/(np.sqrt(c_r_k)+np.finfo(float).eps))
        self.v_k_1 = v_k
        self.r_k_1 = r_k
        self.k =self.k+1

        return weight_tensor_next

