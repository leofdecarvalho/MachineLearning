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
                
#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(x[:, 1:3]) 
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Enconde categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotenconder = OneHotEncoder(categorical_features=[0])
x = onehotenconder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_x.fit_transform(y)

