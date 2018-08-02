# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:38:52 2017

@author: rishi

Simple linear Regression
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3,random_state = 0)

# Feature Scaling will be done by libraries which are used for simple linear regression

#Fitting Simple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)

# Visualising training set result
plt.scatter (X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


#Visualising test  set result
plt.scatter (X_test,Y_test,color='red')
plt.scatter(X_test,y_pred,color='green')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test, y_pred) 





