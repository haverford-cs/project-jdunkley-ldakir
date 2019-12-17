"""
Authors: Lamiaa Dakir
Date:
Description:
"""

import numpy as np
from util1 import *
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from spamFilter import parse_email

def main() :
    #load data
    train_data, test_data = load_data('spambase/spambase.data')

    #Using Adaboost Classifier with 200 classifiers
    print('Using AdaBoost ...')
    clf = AdaBoostClassifier(n_estimators=200, learning_rate = 1)
    clf.fit(train_data.X, train_data.y)

    #Training and Testing Accuracy
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))


    #Creating Confusion Matrix
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

    #Creating a roc curve for different number of classifiers
    clf_200 = AdaBoostClassifier(n_estimators=200, learning_rate = 1)
    clf_200.fit(train_data.X, train_data.y)
    y_score_200 = clf_200.decision_function(test_data.X)
    fpr_200, tpr_200, thresholds_200 = roc_curve(test_data.y, y_score_200)

    clf_500 = AdaBoostClassifier(n_estimators=50, learning_rate = 1)
    clf_500.fit(train_data.X, train_data.y)
    y_score_500 = clf_500.decision_function(test_data.X)
    fpr_500, tpr_500, thresholds_500 = roc_curve(test_data.y, y_score_500)

    clf_20 = AdaBoostClassifier(n_estimators=20, learning_rate = 1)
    clf_20.fit(train_data.X, train_data.y)
    y_score_20 = clf_20.decision_function(test_data.X)
    fpr_20, tpr_20, thresholds_20 = roc_curve(test_data.y, y_score_20)

    #Plotting Roc Curve for T = 20,200 and 500
    plt.plot(fpr_200,tpr_200,'r-', label = 'T= 200')
    plt.plot(fpr_500,tpr_500,'g-', label = 'T= 500')
    plt.plot(fpr_20,tpr_20,'b-', label = 'T= 20')
    plt.legend()
    plt.title("ROC curve for AdaBoost Classifier")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.savefig('adaboost.png')
    plt.show()


    #Finding the features' importance
    feature_names =['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over' , 'word_freq_remove', 'word_freq_internet','word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report',\
    'word_freq_addresses','word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',\
    'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',\
    'char_freq_[', 'char_freq_!','char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
    feature_imp = clf_200.feature_importances_
    print('\n')
    print('Feature names ...')
    print(feature_names)
    print('\n')
    print('Feature importance ...')
    print(feature_imp)
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

    print('\n')
    print('\n')
    #Using RandomForest Classifier with 20 classifiers
    print('Using RandomForest ...')
    clf = RandomForestClassifier(n_estimators=20, criterion = 'gini')
    clf.fit(train_data.X, train_data.y)

    #Training and Testing Accuracy
    print('Training Accuracy: ', clf.score(train_data.X,train_data.y))
    print('Testing Accuracy: ', clf.score(test_data.X,test_data.y))

    prediction = clf.predict(test_data.X)

    #Finding the features' importance
    feature_names =['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over' , 'word_freq_remove', 'word_freq_internet','word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report',\
    'word_freq_addresses','word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',\
    'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',\
    'char_freq_[', 'char_freq_!','char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
    feature_imp = clf.feature_importances_
    print('\n')
    print('Feature names ...')
    print(feature_names)
    print('\n')
    print('Feature importance ...')
    print(feature_imp)
    print('\n')


    #Creating Confusion Matrix
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
