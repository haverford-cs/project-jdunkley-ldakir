"""
Authors: Lamiaa Dakir
Date:
Description:
"""

import numpy as np
from util1 import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from spamFilter import parse_email

def main() :
    #load data
    train_data, test_data = load_data('spambase/spambase.data')

    #Using decision trees classifier
    print('Using Decision Trees ...')

    print('\n')
    print('Params: entropy criterion - best splitter')
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: entropy criterion - random splitter')
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "random")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: gini criterion - best splitter')
    clf = DecisionTreeClassifier(criterion = "gini", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    print('\n')
    print('Params: gini criterion - random splitter')
    clf = DecisionTreeClassifier(criterion = "gini", splitter = "random")
    clf = clf.fit(train_data.X, train_data.y)
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))


    #Testing the accuracy of the decision tree classifier in terms of the maximum depth
    print('\n')
    print('Params: entropy criterion - best splitter - max depth')
    training_accuracies = []
    testing_accuracies = []
    max_depth =[]
    for i in range(1,50):
        max_depth.append(i)
        clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth = i)
        clf = clf.fit(train_data.X, train_data.y)
        training_accuracy = clf.score(train_data.X,train_data.y)
        testing_accuracy = clf.score(test_data.X,test_data.y)
        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)
        print('Max Depth: ', i)
        print('Training Accuracy: ',training_accuracy)
        print('Testing Accuracy: ',testing_accuracy)

    plt.plot(max_depth,training_accuracies,'bo-',label ='training accuracy')
    plt.plot(max_depth,testing_accuracies,'ro-',label ='testing accuracy')
    plt.title("Accuracy of Decision Trees vs Maximum Depth of the Tree")
    plt.legend()
    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")
    plt.savefig('DecisionTreesAccuracy')
    plt.show()

    #Creating a confusion matrix for the classifier using the entropy criterion and splitting by best feature
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
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
    print('   -1  1')
    print('   -----')
    print('-1| '+ str(int(confusion_matrix[0][0])) + '  ' + str(int(confusion_matrix[0][1])))
    print(' 1| '+ str(int(confusion_matrix[1][0])) + '  ' + str(int(confusion_matrix[1][1])))
    print('\n')

    #Visualizing decision tree with depth 2 to see the best features
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth=2)
    clf = clf.fit(train_data.X, train_data.y)
    prediction = clf.predict(test_data.X)
    plot_tree(clf, feature_names =['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over' , 'word_freq_remove', 'word_freq_internet','word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report',\
    'word_freq_addresses','word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',\
    'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',\
    'char_freq_[', 'char_freq_!','char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'], filled = True)
    plt.savefig('decisionTree.png')
    plt.show()

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
