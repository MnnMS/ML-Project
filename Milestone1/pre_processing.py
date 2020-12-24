import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics


def One_Hot_Encoding(X,cols):
    for c in cols:
        # get one hot encoding for X[c] column
        df2 = pd.get_dummies(X[c])
        #drop X[c] column
        X.drop(columns=[c],axis=1, inplace=True)
        #add new columns into df
        X = pd.concat([X, df2], axis=1,sort= False)
    return X

def featureScaling(X,a,b):
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        mx = max(X[:,i])
        mn = min(X[:,i])
        if mx != mn:
            Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(mx-mn))*(b-a)+a
    return Normalized_X