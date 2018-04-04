# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer().fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
x[:, 0] = LabelEncoder().fit_transform(x[:, 0])
x = OneHotEncoder(categorical_features = [0]).fit_transform(x).toarray()
y = LabelEncoder().fit_transform(y)

# Spilit the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

