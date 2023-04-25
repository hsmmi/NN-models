import numpy as np


class activation_func:
    def __init__(self, mode):
        self.mode = mode

    def sign(x, theta=0):
        """
        Sign function
        """
        if x > theta:
            return 1
        elif x < theta:
            return -1
        else:
            return 0

    def step_binary(x, theta=0):
        """
        Step function
        """
        if x > theta:
            return 1
        else:
            return 0

    def step_bipolar(x, theta=0):
        """
        Step function
        """
        if x > theta:
            return 1
        else:
            return -1

    def binary_sigmoid(x):
        """
        Binary sigmoid function
        """
        return 1 / (1 + np.exp(-x))
    
    def bipolar_sigmoid(x):
        """
        Bipolar sigmoid function
        """
        return 2 / (1 + np.exp(-x)) - 1
    
    def d_binary_sigmoid(x):
        """
        Derivative of binary sigmoid function
        """
        return activation_func.binary_sigmoid(x) * (1 - activation_func.binary_sigmoid(x))
    
    def d_bipolar_sigmoid(x):
        """
        Derivative of bipolar sigmoid function
        """
        return (1 + activation_func.bipolar_sigmoid(x)) * (1 - activation_func.bipolar_sigmoid(x))
        return 0.5 * (1 + activation_func.bipolar_sigmoid(x)) * (1 - activation_func.bipolar_sigmoid(x))
    
    def identity(x):
        """
        Identity function
        """
        return x
    
    def d_identity(x):
        """
        Derivative of identity function
        """
        return 1