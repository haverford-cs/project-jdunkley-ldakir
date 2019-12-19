"""
Pre-processing the data for the FC NN
Authors: Jocelyn Dunkley and Lamiaa Dakir
Date: 12/04/19
"""

# python imports
import os
import numpy as np
from math import sqrt, exp, log
from random import randrange

# my file imports
class Data :

    def __init__(self, data = None, X=None, y=None) :
        """
            X       -- numpy array of shape (n,p), features
            y       -- numpy array of shape (n,), targets
        """
        # n = number of examples, p = dimensionality
        self.data = data
        self.X = X
        self.y = y


    def load(self, filename) :
        """
        Load file into X array of features and y array of labels.
        filename (string)
        """
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        #train_data, test_data = self.split_train_test(data)

        # separate features and labels
        #self.X = data[:,:-1]
        #self.y = data[:,-1]
        self.data = data


def split_train_test(data):
    train_data =[]
    test_data =[]
    i = len(data)*0.75
    while len(data) != 0:
        rand = randrange(len(data))
        if i > 0:
            train_data.append(data[rand,:])
        else:
            test_data.append(data[rand,:])

        data = np.delete(data,rand,0)
        i -=1
    return train_data, test_data



def load_data(filename) :
    data = Data()
    data.load(filename)

    #then you divide it up here
    train_data, test_data = split_train_test(data.data)
    print(len(train_data), len(test_data))

    return train_data, test_data
