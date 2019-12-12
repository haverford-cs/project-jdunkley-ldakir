"""
Fully connected neural network architecture.
Author:
Date:
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

##################

class FCmodel(Model):
    """
    A fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(FCmodel, self).__init__()
        #Flattening
        self.flatten = Flatten()
        #Creating layers
        self.d1 = Dense(4000, activation=tf.nn.relu)
        self.d2 = Dense(2, activation=tf.nn.softmax) #57

        # TODO set up architecture, for example:
        #self.d1 = Dense(<num hidden units>, <activation function>)
        # use 4000 units in the hidden layer, num classes is 10
    #x is the mini-batch of pixels ?? (64,32,32,3)
    def call(self, x):

        # TODO apply each layer from the constructor to x, returning
        # the output of the last layer
        x1 = self.flatten(x)
        #return self.flatten
        x2 = self.d1(x1)
        return self.d2(x2)

def two_layer_fc_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    fc_model = FCmodel()

    # TODO try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    #x_np = np.zeros((64, 32, 32, 3))
    x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = fc_model.call(x_np)
    print("output! ", output) # TODO what shape is this? does it make sense?
    #shape=(64, 32, 32, 10)

    """
    shape=(64, 10), dtype=float32)
    Variable: dense/kernel:0
    Shape: (3072, 4000)
    Variable: dense/bias:0
    Shape: (4000,)
    Variable: dense_1/kernel:0
    Shape: (4000, 10)
    Variable: dense_1/bias:0
    Shape: (10,)
    """
    # TODO look at the model parameter shapes, do they make sense?
    for v in fc_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test two layer function
    two_layer_fc_test()

if __name__ == "__main__":
    main()
