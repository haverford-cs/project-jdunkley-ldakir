"""
Authors: Lamiaa Dakir and Jocelyn Dunkley
Date: 12/17/2019
Description: Util file to load data as features and labels.
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
        self.X = X
        self.y = y



def load(filename) :
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

    return data

def split_train_test(data):
    train_split =[]
    test_split =[]
    i = len(data)*0.75

    while len(data) != 0:
        #print(len(data))
        rand = randrange(len(data))
        if i > 0:
            train_split.append(data[rand,:])
        else:
            test_split.append(data[rand,:])

        data = np.delete(data,rand,0)
        i -=1

    train_split = np.array(train_split)
    test_split = np.array(test_split)


    train_data = Data()
    train_data.X  = train_split[:,:-1]
    train_data.y  = train_split[:,-1]

    test_data = Data()
    test_data.X  = test_split[:,:-1]
    test_data.y  = test_split[:,-1]

    return train_data, test_data

def load_data(filename) :
    data = load(filename)
    train_data, test_data = split_train_test(data)
    return train_data, test_data
