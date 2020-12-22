import pandas as pd
import numpy as np
def One_Hot_Encoding(X,cols):
    for c in cols:
        # get one hot encoding for X[c] column
        df2 = pd.get_dummies(X[c])
        #drop X[c] column
        X.drop(c,axis=1, inplace=True)
        #add new columns into df
        X = pd.concat([X, df2], axis=1)
    return X

def featureScaling(X,a,b):
    Normalized_X=np.zeros((X.shape[0],X.shape[1]));
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a;
    return Normalized_X