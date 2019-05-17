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

    k = KNeighborsClassifier(n_neighbors = 3)
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

    print("Hello World")
    
    # Just testing with parkinsons Data atm
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

    # Process data
    data = generateDataSetFromUrl(url)
    target = data['status']
    data = removeNames(data)

    # Splitting data into training and testing data ( Will go with 80:20 approach considering we don't have too much data with this set)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2, random_state = 10)

    # Running Guassian Naive Bayes
    print("Running Naive-Bayes....\n")
    gnbPredicition = guassianNaiveBayes(data_train, target_train, data_test)
    print("Naive-Bayes accuracy : " + str(accuracy_score(target_test, gnbPredicition, normalize = True)) + "\n")

    # Running Support Vector Classification
    print("Running Support Vector Classification....\n")
    svcPredicition = supportVectorClassification(data_train, target_train, data_test)
    print("Support Vector Classification accuracy : " + str(accuracy_score(target_test, svcPredicition, normalize = True)) + "\n")

    # Running k nearest neighbors
    print("Running K Neighbors Classifier....\n")
    kNearPredicition = kNeighbors(data_train, target_train, data_test)
    print("K Neighbor Classifier accuracy : " + str(accuracy_score(target_test, kNearPredicition, normalize = True)) + "\n")

    # Running Decision Tree
    print("Running Decision Tree Classifier....\n")
    dTreePredicition = decisionTree(data_train, target_train, data_test)
    print("Decision Tree Classifier accuracy : " + str(accuracy_score(target_test,dTreePredicition, normalize = True)) + "\n")


if __name__ == "__main__":
    main()