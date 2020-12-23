from pre_processing import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
#import poly_model as pm

data = pd.read_csv('AppleStore_training.csv')
data.dropna(how='any',inplace=True)
X=data.iloc[:,[2,8]]
Y=data['user_rating']
# cols=['cont_rating']
# X = One_Hot_Encoding(X,cols)
X = featureScaling(np.array(X),0,1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=False)
poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
#y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
X_test_poly = poly_features.fit_transform(X_test)
prediction = poly_model.predict(X_test_poly)

print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))