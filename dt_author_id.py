#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
    
import sys
from time import time
from simpsons_script_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', max_features=250, max_depth=50)

# reduce the size of the data set for speed
features_train = features_train[:len(features_train)/20] 
labels_train = labels_train[:len(labels_train)/20] 
clf.fit(features_train, labels_train)

print "Decision Tree Accuracy:", clf.score(features_test, labels_test)

#########################################################


