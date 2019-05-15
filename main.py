''' Machine Learning Final Project '''

# Some necessary import statements
import os
import time
import sys
import numpy as np
import pandas as pd
import urllib 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

    # Running Guassian Naive Bayes
    gnbPredicition = guassianNaiveBayes(data_train, target_train, data_test)
    print("Naive-Bayes accuracy : " + str(accuracy_score(target_test, gnbPredicition, normalize = True)))

if __name__ == "__main__":
    main()