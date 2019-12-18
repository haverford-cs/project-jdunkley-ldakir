# Jocelyn Dunkley & Lamiaa Dakir

# Is It A Spam ?
Our final project aims to evaluate which algorithm would be the most accurate to determine if an email is spam or not. We implemented Decision Trees, Random Forests, AdaBoost, Naive Bayes and Fully Connected Neural Networks. It turned out that Random Forests and AdaBoost had the best/comparable accuracies but AdaBoost correctly predicted a real spam email while Random Forests classifed it as not spam. 

# Data Pre-processing

Link to the dataset: https://archive.ics.uci.edu/ml/datasets/spambase.

The data has 57 features and one label. It has 4601 examples, 1813 of which are spam and 2788 are non-spam. During the preprocessing, we split the data into 75% training data and 25% testing data. We then separate the features from the labels and make sure we do not look at the testing data labels. To ensure the training data and testing data is not biased we randomly choose points from the dataset.

# Decision Trees
Using the DecisionTreeClassifier in the sklearn.tree python package, we fit the training data and evaluate the performace of the decision tree model.
    
    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)
    
The decision tree classifier achieved a higher accuracy using the entropy criterion and the best splitter. To visualize the best features choosen, we plot the decision tree of a maximum depth of 2.

    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth=2)
    clf = clf.fit(train_data.X, train_data.y)
    plot_tree(clf)
    
# AdaBoost and Random Forest
We use the AdaBoostClassifier and RandomForestClassifier in the sklearn.ensemble python package. The adaBoost model was able to achieve a higher accuracy with 200 estimators while Random Forest only required about 20 estimators. The two classifiers choose the importance of the features differently. Random Forest gives a higher importance to some features compared to others while AdaBoost gives about the same importance values to all the features.

    clf = AdaBoostClassifier(n_estimators=200, learning_rate = 1)
    clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)

    clf = RandomForestClassifier(n_estimators=20, criterion = 'gini')
    clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)
    
# Naive Bayes
We use GaussianNB in the sklearn.naive_bayes python package. GuassianNB handles continuous features and can be used in this case because the features are independent.

    clf = GaussianNB()
    clf.fit(train_data.X, train_data.y)
    
Naive Bayes is very strict because the false negative value in the confusion matrix is very high. It is more likely to classify an email as a spam.

  
   
    


Our Lab Notebook/Log is in a Google doc that can be accessed here: https://docs.google.com/document/d/1bO7JDyPeJD8Q4W5iTv7HU_1woeriWaAzBNPrrJdDe-E/edit?usp=sharing
