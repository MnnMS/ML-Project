import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
poly_model = linear_model.LinearRegression()
def train(X,Y):
    poly_model.fit(X, Y)

def test(X, Y):
    prediction = poly_model.predict(poly_features.fit_transform(X))
    return prediction



