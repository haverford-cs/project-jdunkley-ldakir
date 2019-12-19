# Jocelyn Dunkley & Lamiaa Dakir

# Is It A Spam ?
Our final project aims to evaluate which algorithm would be the most accurate to determine if an email is spam or not. We implemented Decision Trees, Random Forests, AdaBoost, Naive Bayes and Fully Connected Neural Networks. It turned out that Random Forests and AdaBoost had the best/comparable accuracies but AdaBoost correctly predicted a real spam email while Random Forests classified it as not spam.

# Contents of the Repo:
1. Spambase dataset
2. Our original project proposal
3. Our presentation slides to demonstrate our results. We gave the presentation on December 12th.
4. The implementation our different algorithms: Decision Trees, AdaBoost, Random Forests, Naive Bayes and Neural Networks
5. spamFilter.py: testing our algorithms on two different emails - one spam (antispamSpam.txt), one not-spam (saraEmail.txt) - to test if it could correct classify spam from non-spam.
6. Two util files to preprocess our data. One is for the Neural Networks and the other is for the rest of the algorithms, since the process was slightly different.
7. run_data.py: To test if loading/preprocessing the data is working.

# Lab Notebook
Our Lab Notebook/Log is in a Google doc that can be accessed here: https://docs.google.com/document/d/1bO7JDyPeJD8Q4W5iTv7HU_1woeriWaAzBNPrrJdDe-E/edit?usp=sharing

# Data Pre-processing

Link to the dataset: https://archive.ics.uci.edu/ml/datasets/spambase.

The data has 57 continuous features and one label. It has 4601 examples, 1813 of which are spam and 2788 are non-spam, which is an approximate 60% non-spam/40% spam split. During preprocessing, we split the data into 75% training data and 25% testing data. We then separated the features from the labels and made sure we did not look at the testing data labels. To ensure the training data and testing data was not biased we randomly chose points from the dataset, since the dataset was organized in a way where all the points with label = 1 were listed and then all the points with label = 0.

# Decision Trees
Using the DecisionTreeClassifier in the sklearn.tree python package, we fitted the training data and evaluated the performance of the decision tree model.

    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best")
    clf = clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)

The decision tree classifier achieved a higher accuracy using the entropy criterion and the best splitter. To visualize the best features chosen, we plotted the decision tree of a maximum depth of 2.

    clf = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_depth=2)
    clf = clf.fit(train_data.X, train_data.y)
    plot_tree(clf)

# AdaBoost and Random Forest
We used the AdaBoostClassifier and RandomForestClassifier in the sklearn.ensemble python package. The adaBoost model was able to achieve a higher accuracy with 200 estimators while Random Forest only required about 20 estimators. The two classifiers chose the importance of the features differently. Random Forest gave a higher importance to some features compared to others while AdaBoost gave about the same importance values to all the features.

    clf = AdaBoostClassifier(n_estimators=200, learning_rate = 1)
    clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)

    clf = RandomForestClassifier(n_estimators=20, criterion = 'gini')
    clf.fit(train_data.X, train_data.y)
    score = clf.score(test_data.X,test_data.y)

# Naive Bayes
We used GaussianNB in the sklearn.naive_bayes python package. GuassianNB handles continuous features and can be used in this case because the features are independent.

    clf = GaussianNB()
    clf.fit(train_data.X, train_data.y)

Naive Bayes was very strict because the false negative value in the confusion matrix was very high. It is more likely to classify an email as a spam.

# Fully Connected Neural Network
We used the tf.keras API in Tensorflow to easily build our Fully Connected Neural Network. The first layer Flattens and takes an initial size of 57, since that is the number of features. Then there is a Dense layer with 100 hidden nodes using ReLU. Note: I learned that you want to have the number of hidden nodes usually be a little more than half the number of your examples, I was originally using 4000, which was way too high and complicated for my model. Then I used Batch Normalization and a Dropout of 0.2 then an output Dense layer with size 2 (since there are two labels) using softmax. We added an adam optimizer, sparse_categorical_crossentropy loss function and an accuracy metric.

Using Batch Normalization and Dropout helped to make the accuracies pretty consistent around 90-92% on testing data.

    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(57,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

The NN turned out to be quite lenient when classifying. It had one of the higher false negative rates, meaning that it would classify an email as not spam when it was actually spam.
