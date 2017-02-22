#Data Pre Processing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#print all data
np.set_printoptions(threshold=np.inf)

#Import the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values   

# Splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_text = train_test_split(x, y,test_size= 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
