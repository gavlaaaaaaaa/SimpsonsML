#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)

print "Naive Bayes accuracy:", clf.score(features_test, labels_test)

#########################################################

from sklearn.svm import SVC
clf = SVC(kernel='linear')
features_train = features_train[:len(features_train)/60] 
labels_train = labels_train[:len(labels_train)/60] 
clf.fit(features_train, labels_train)

print "SVM Accuracy:", clf.score(features_test, labels_test)

