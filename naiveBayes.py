"""
Authors: Lamiaa Dakir
Date:
Description:
"""

import numpy as np
from util1 import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import matplotlib.pyplot as plt
from spamFilter import parse_email

def main() :
    #load data
    train_data, test_data = load_data('spambase/spambase.data')

    #Using Gaussian Naive Bayes classifier
    print('Using Naive Bayes ...')
    clf = GaussianNB()
    clf.fit(train_data.X, train_data.y)

    #Testing and Training Accuracies
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))


    #Creating a confusion matrix for the classifier
    prediction = clf.predict(test_data.X)
    confusion_matrix = np.zeros((2,2))
    accuracy =0
    for i in range(len(prediction)):
        if  prediction[i] == 0 and test_data.y[i] == 0:
            confusion_matrix[0][0] +=1
            accuracy +=1
        elif prediction[i] == 1 and test_data.y[i] == 1:
            confusion_matrix[1][1] +=1
            accuracy +=1
        elif prediction[i] == 0 and test_data.y[i]  == 1:
            confusion_matrix[1][0] +=1

        elif prediction[i] == 1 and test_data.y[i]  == 0:
            confusion_matrix[0][1] +=1

    #Outputting confusion matrix
    print('\n')
    print('Confusion Matrix')
    print(' prediction')
    print('   0  1')
    print('   -----')
    print(' 0| '+ str(int(confusion_matrix[0][0])) + '  ' + str(int(confusion_matrix[0][1])))
    print(' 1| '+ str(int(confusion_matrix[1][0])) + '  ' + str(int(confusion_matrix[1][1])))
    print('\n')


    #Using the email parser can we detect if the email is spam ?
    print('\nParsing spam email...')
    email = Data()
    email.X, email.y = parse_email('antispamSpam.txt',1)
    prediction = clf.predict(email.X)[0]
    true_label = email.y[0]
    print(prediction, true_label)
    if prediction == true_label:
        print('Successfully detected the spam.')
    else:
        print('Failed to detect the spam.')


    #Using the email parser can we detect if the email is not spam ?
    print('\nParsing Sara\'s email...')
    not_spam = Data()
    not_spam.X, not_spam.y = parse_email('saraEmail.txt',0)
    prediction = clf.predict(not_spam.X)[0]
    true_label = not_spam.y[0]
    print(prediction, true_label)
    if prediction == true_label:
        print('Successfully detected that the email is safe.')
    else:
        print('Misclassified the email as spam.')




if __name__ == "__main__" :
    main()
