## Created by: Zain Alsaad

## Resources/Refrences used:
#https://colab.research.google.com/drive/1xBqbJfMASv-rjc0Hvbj-PI0qMQM3WqRs?usp=sharing
#https://colab.research.google.com/drive/1QzyXP1sKBIOuz46f8t-WVYR2bKDgASff?usp=sharing

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Cross-validate by 10 and print F1, accuracy, precision, and recall performance of classifier parameter
def crossVal(classifier, X_train, y_train, X_test, y_test):
    # Fit the classifier first
    classifier.fit(X_train, y_train)

    # Cross-validation scores
    f1 = cross_val_score(classifier, X_train, y_train, scoring='f1_weighted', cv=10)
    accuracy = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
    precision = cross_val_score(classifier, X_train, y_train, scoring='precision', cv=10)
    recall = cross_val_score(classifier, X_train, y_train, scoring='recall', cv=10)

    print("F1: %.2f%%" % (100 * f1.mean()))
    print("Accuracy: %.2f%%" % (100 * accuracy.mean()))
    print("Precision: %.2f%%" % (100 * precision.mean()))
    print("Recall: %.2f%%" % (100 * recall.mean()))

    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)[:, 1]
    else:
        clf = CalibratedClassifierCV(classifier, cv="prefit")
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]


##AUROC EXTRA CREDIT
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print("AUC: %.2f" % roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    # Path to the file containing all of the comments
    path = 'icpc17'

    # Where the comments will be stored
    concernFiles = []
    miscfiles = []  # other comments
    allComments = []  # all comments combined

    # Loop through all the files and get the concerns and other txt/comments files
    for file in os.listdir(path):
        if file != '.DS_Store':
            with open(path + '/' + file + '/concerns.txt') as fil:
                for line in fil.readlines():
                    concernFiles.append(line)
                    allComments.append(line)

            with open(path + '/' + file + '/others.txt') as fil:
                for line in fil.readlines():
                    miscfiles.append(line)
                    allComments.append(line)

    # Encode the target labels.
    zeros = np.zeros(len(concernFiles))  # Concern comments are 0
    ones = np.ones(len(miscfiles))  # Miscellaneous comments are 1

    # Creates the feature matrix and label vector.
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(allComments)
    y = np.append(zeros, ones)

    # Create the tf-idf.
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

    # Initialize classifiers
    SVMclassifier = LinearSVC(random_state=0)
    NaiveBayesclassifier = MultinomialNB()
    DecisionTree = DecisionTreeClassifier(random_state=0, max_depth=5)
    RandomForest = RandomForestClassifier(random_state=0, max_depth=5)

    # Print each classifier and their scores
    print('SVM Classifier Scores:')
    crossVal(SVMclassifier, X_train, y_train, X_test, y_test)

    print('Naive Bayes Classifier Scores:')
    crossVal(NaiveBayesclassifier, X_train, y_train, X_test, y_test)

    print('Decision Tree Classifier Scores:')
    crossVal(DecisionTree, X_train, y_train, X_test, y_test)

    print('Random Forest Classifier Scores:')
    crossVal(RandomForest, X_train, y_train, X_test, y_test)
