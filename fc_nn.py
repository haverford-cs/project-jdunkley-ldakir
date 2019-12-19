"""
Loading and building FC NN model for the spambase dataset
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Jocelyn Dunkley
Date: 12/11/19
"""

import numpy as np
import os
import tensorflow as tf

from tensorflow.python.keras import backend as K
from util_fc_nn import *

def load_spambase(num_training=3451, num_validation=1000, num_test=1150):
    #3451 1150
    """
    Fetch the spambase dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier.
    """
    train_data, test_data = load_data('spambase/spambase.data')
    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    X_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    # Subsample the data for validation
    mask = range(num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_spambase()
    print('Train data shape: ', X_train.shape)              #(3451, 57)
    print('Train labels shape: ', y_train.shape)            #(3451,)
    print('Validation data shape: ', X_val.shape)           #(1000, 57)
    print('Validation labels shape: ', y_val.shape)         #(1000,)
    print('Test data shape: ', X_test.shape)                # (1150, 57)
    print('Test labels shape: ', y_test.shape)              # (1150,)


    buffer = X_train.shape[0]

    #prepare train and test dset for tensorflow functions
    train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(buffer).batch(64)
    test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

    #setting up the architecture
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(57,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
    ])
    #adding optimizers and loss functions
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    #training the model
    model.fit(X_train, y_train, epochs=10)
    #testing the model
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    #making the confusion matrix
    predictions = tf.argmax(model.predict(X_test), axis=1, output_type=tf.int32)
    print("confusion matrix", tf.math.confusion_matrix(y_test, predictions))


main()
