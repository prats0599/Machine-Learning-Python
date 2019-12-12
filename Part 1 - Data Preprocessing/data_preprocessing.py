#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

#taking care of missing data by replacing it with mean of data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose = 0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

#encoding categorical data
#columns 0, 1, 2 are france, germany and spain respectively. for y vector= 0-->no 1--> yes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = pd.DataFrame(X)
labelencoder_Y = LabelEncoder()
Y.iloc[:, 0] = labelencoder_Y.fit_transform(Y)

#spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#object sc_X is already fitted to X_train here so just use tranform for X_test.
X_test = sc_X.transform(X_test)



