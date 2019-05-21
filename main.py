''' Machine Learning Final Project '''

# Some necessary import statements
import os
import time
import sys
import numpy as np
import pandas as pd
import urllib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Imports for machine learning algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Process url into comprehensive 2D datatset
def generateDataSetFromUrl(url):
    return pd.read_csv(url)

# Remove unecessary info and assign a target variable
def removeNames(dataset):
    cols = []
    for col in dataset.columns:
        if col not in ['name', 'status']:
            cols.append(col)

    return dataset[cols]

# Running training data through a guassian naive model generator
def guassianNaiveBayes(dtrain, ttrain, dtest):

    gnb = GaussianNB()
    prediction = gnb.fit(dtrain, ttrain).predict(dtest)
    return prediction

# Running training data through a SVC model
def supportVectorClassification(dtrain, ttrain, dtest):

    svc = LinearSVC(random_state = 10, max_iter = 200000)
    prediction = svc.fit(dtrain, ttrain).predict(dtest)
    return prediction

# Running training data through a k neighbors model
def kNeighbors(dtrain, ttrain, dtest):

    k = KNeighborsClassifier(n_neighbors = 3) # used n_neighbors = 3
    k.fit(dtrain, ttrain)
    prediction = k.predict(dtest)
    return prediction

# Running training data through decision tree classifier
def decisionTree(dtrain, ttrain, dtest):

    dTree = DecisionTreeClassifier()
    dTree = dTree.fit(dtrain, ttrain)
    prediction = dTree.predict(dtest)
    return prediction

# Main function declaration
def main():

    # Just testing with parkinsons Data atm
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

    # Process data
    data = generateDataSetFromUrl(url)
    target = data['status']
    data = removeNames(data)

    # Splitting data into training and testing data ( Will go with 80:20 approach considering we don't have too much data with this set)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2, random_state = 10)

    # Print info about the dataset
    print(str(len(data))       + " elements in the total set")
    print(str(len(data_train)) + " elements in the training set")
    print(str(len(data_test))  + " elements in the testing set")
    print() # Prints an empty line of spacing

    # Running Guassian Naive Bayes
    start = time.time()
    print("Running Naive-Bayes....")
    gnbPredicition = guassianNaiveBayes(data_train, target_train, data_test)
    gnbAccuracy = accuracy_score(target_test, gnbPredicition, normalize = True)
    #print("Naive-Bayes accuracy : " + str(accuracy_score(target_test, gnbPredicition, normalize = True)))
    print("Naive-Bayes accuracy : " + str(gnbAccuracy))
    end = time.time()
    print("     " + str(round(gnbAccuracy * len(data_test))) + " correctly classified out of " + str(len(data_test)))
    print("Total time: " + str(end - start) + "\n")

    # Running Support Vector Classification
    #start = time.time();
    #print("Running Support Vector Classification....")
    #svcPredicition = supportVectorClassification(data_train, target_train, data_test)
    #print("Support Vector Classification accuracy : " + str(accuracy_score(target_test, svcPredicition, normalize = True)))
    #end = time.time();
    #print("Total time: " + str(end - start) + "\n")

    # Running k nearest neighbors
    start = time.time()
    print("Running K Neighbors Classifier....")
    kNearPredicition = kNeighbors(data_train, target_train, data_test)
    kNearAccuracy = accuracy_score(target_test, kNearPredicition, normalize = True)
    #print("K Neighbor Classifier accuracy : " + str(accuracy_score(target_test, kNearPredicition, normalize = True)))
    print("K Neighbor Classifier accuracy : " + str(kNearAccuracy))
    end = time.time()
    print("     " + str(round(kNearAccuracy * len(data_test))) + " correctly classified out of " + str(len(data_test)))
    print("Total time: " + str(end - start) + "\n")

    # Running Decision Tree
    start = time.time();
    print("Running Decision Tree Classifier....")
    dTreePredicition = decisionTree(data_train, target_train, data_test)
    dTreeAccuracy = accuracy_score(target_test,dTreePredicition, normalize = True)
    #print("Decision Tree Classifier accuracy : " + str(accuracy_score(target_test,dTreePredicition, normalize = True)))
    print("Decision Tree Classifier accuracy : " + str(dTreeAccuracy))
    end = time.time();
    print("     " + str(round(dTreeAccuracy * len(data_test))) + " correctly classified out of " + str(len(data_test)))
    print("Total time: " + str(end - start) + "\n")


if __name__ == "__main__":
    main()