import time
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def multivariable_reg(x, y, Reg=True):
    if not Reg:
        model = linear_model.LinearRegression()
    else:
        model = linear_model.Ridge()
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def polynomial_reg(x, y, deg=3, Reg=True):
    poly_features = PolynomialFeatures(degree=deg)
    X_train_poly = poly_features.fit_transform(x)
    if not Reg:
        model = linear_model.LinearRegression()
    else:
        model = linear_model.Ridge(alpha=1e-5)
    start = time.time()
    model.fit(X_train_poly, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime