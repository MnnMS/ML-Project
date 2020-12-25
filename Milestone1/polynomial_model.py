import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from pre_processing import *


def pre_process(data):
    data.drop(columns=['currency', 'ver', 'id'], inplace=True)
    # all_data = data.iloc[:,:]
    X = data.iloc[:, 1:12]
    cols = ['prime_genre', 'cont_rating']
    return X, cols

def train(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)


    poly_features = PolynomialFeatures(degree=3)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
   # poly_model = linear_model.LinearRegression()
    poly_model = Ridge(alpha=1e-5)
    poly_model.fit(X_train_poly, y_train)

    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    return prediction, y_test

