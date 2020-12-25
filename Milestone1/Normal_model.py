import numpy as np



def normalEquation(X, y):
    step1 = np.dot(X.T, X)
    step2 = np.linalg.pinv(step1)
    step3 = np.dot(step2, X.T)
    theta = np.dot(step3, y)
    return theta
def predict(X,Theta):
    prediction = X.dot(Theta)
    return prediction

def pre_process(data):
    data.drop(columns=['prime_genre','cont_rating', 'track_name','ver', 'currency'], inplace=True)
    X=data.iloc[:,0:10]

    cols=[]
    return X,cols

def fit(X,Y):
    Y = np.expand_dims(Y, axis=1)
    X = np.c_[np.ones((len(X), 1)), X]
    Theta = normalEquation(X, Y)
    prediction = predict(X, Theta)
    return prediction
