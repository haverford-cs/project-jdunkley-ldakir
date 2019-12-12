"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors:
Date:
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

from fc_nn import *
from util import *

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


def run_training(model, train_dset, val_dset):

    confusion_matrix = np.zeros((10,10))
    training_acc_epoch = []
    validation_acc_epoch = []
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    #model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
    # use the Adam optimizer
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Prepare the metrics.
    #Calculates how often predictions matches integer labels.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    epochs = 10
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
      # Iterate over the batches of the dataset.
        #training I guess
        for step, (x_batch_train, y_batch_train) in enumerate(train_dset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:

              # Run the forward pass of the layer.
              # The operations that the layer applies
              # to its inputs are going to be recorded
              # on the GradientTape.
              logits = model.call(x_batch_train)  # Logits for this minibatch

              # Compute the loss value for this minibatch.
              loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            adam_optimizer.apply_gradients(zip(grads, model.trainable_weights))

             # Update training metric.
            train_acc_metric(y_batch_train, logits)
            train_loss(loss_value)
            # Log every 200 batches.
            if step % 64 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        training_acc_epoch.append(float(train_acc))
        print('Training acc over epoch: %s' % (float(train_acc),))
        print('Training loss over epoch: ', float(train_loss.result()))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dset:
            val_logits = model.call(x_batch_val)
            # Update val metrics
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        #validation_acc_epoch.append(float(val_acc))
        val_acc_metric.reset_states()
        print('Validation acc: %s' % (float(val_acc),))
        validation_acc_epoch.append(float(val_acc))

    return training_acc_epoch, validation_acc_epoch


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_spambase()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)                # (1150, 57)
    print('Test labels shape: ', y_test.shape)              # (1150,)


    buffer = X_train.shape[0]


    train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    train_dset = train_dset.shuffle(buffer)
    train_dset = train_dset.batch(64)
    val_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)


    """for images,labels in test_dset:
       print("images.shape: ", images.shape)
       print("labels.shape: ", labels.shape)"""

    print("RUNNING FC MODEL")
    fc_model = FCmodel()
    training_acc_fc, validation_acc_fc = run_training(fc_model, train_dset, val_dset)

    #TESTING
    prediction = tf.argmax(fc_model.call(X_test), axis=1, output_type=tf.int32)
    confusion_matrix = np.zeros((2,2))
    test_accuracy = tf.keras.metrics.Accuracy()
    test_accuracy(prediction, y_test)


    print("TRAINING ACCURACY FC")
    print(training_acc_fc)
    print("VALIDATION ACCURACY FC")
    print(validation_acc_fc)
    print("TEST ACCURACY FC")
    print("{:.3%}".format(test_accuracy.result()))
    #print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("confusion matrix", tf.math.confusion_matrix(y_test, prediction))


main()
