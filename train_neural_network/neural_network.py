import numpy as np
from activators import SigmoidActivator


class FullyConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        constructor
        :param input_size: 
        :param output_size: 
        :param activator: 
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # vector of weights W
        # if current layer has 3 neurons and next layer has 2 neurons
        # the weight vector look like:
        # [[w41, w42, w43],
        #  [w51, w52, w53]]
        # and input_size = # of neurons in current layer
        # looks like:
        #        [[x1],
        #         [x2],
        #         [x3]]
        # and the output = W * x + bi
        self.W = np.random.uniform(
            -0.1, 0.1,
            (output_size, input_size)
        )
        # bias for each output term
        # np.zeros((x, y)) return x * y dimension matrix of zeros
        self.b = np.zeros((output_size, 1))

        # output
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        forward calculation for current layer
        :param input_array: vector of input, its dimension must be equal to input_size 
        :return: 
        '''
        self.input = input_array
        weighted_input = np.dot(self.W, input_array)
        self.output = self.activator.forward(weighted_input) + self.b

    def backward(self, delta_array):
        '''
        backpropagation to calculate gradient of W and b
        :param delta_array: the vector delta_j calculated and sent from the next layer
        :return: 
        '''
        # calculate current layer's delta
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T * delta_array)
        self.W_gradient = np.dot(delta_array, self.input.T)
        self.b_gradient = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_gradient
        self.b += learning_rate * self.b_gradient


class Network(object):
    def __init__(self, ):


