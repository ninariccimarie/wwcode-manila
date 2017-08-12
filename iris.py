#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:28:57 2017

@author: ninaricci
"""

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris_dataset = './iris.csv'
df = pd.read_csv(iris_dataset)

iris_df = df.copy()

"""
print('shape',iris_df.shape)
print('head',iris_df.head(10))
print('tail',iris_df.tail(10))
print('data types', iris_df.dtypes)
print('describe', iris_df.describe())

print(iris_df.groupby('Species').size())

print(iris_df.corr(method='pearson'))
"""

X = iris_df.iloc[ :, :4]
y = iris_df.iloc[ :, 4]

train_size = 0.8
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state = seed)

#Instantiate learning model
clf = GaussianNB()

#Fit model to training set
clf.fit(X_train, y_train)

#Predict labels of the test set
y_pred = clf.predict(X_test)

results = pd.DataFrame({'Predicted label:': y_pred, 'True label': y_test})
print(results)

accuracy = accuracy_score(y_test, y_pred) # or accuracy = np.mean(y_pred == y_test)
print('Gaussian Naive Bayes: ', accuracy)

# Or, more concisely (predicts y_pred and evaluates accuracy in a single line of code)
accuracy = clf.score(X_test, y_test)
print('Gaussian Naive Bayes: ',accuracy)