from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import time

def decision_tree(x, y, depth=3):
    model = DecisionTreeClassifier(max_depth=depth)
    start = time.time()
    model.fit(x,y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def adaboost(x,y,depth=3):
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                       algorithm="SAMME",
                       n_estimators=100)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def logistic_reg(x, y):
    model = LogisticRegression()
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def KNN(x, y, n=40):
    model = KNeighborsClassifier(n_neighbors=n)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def svm_rbf(x, y, c):
    model = svm.SVC(kernel='rbf', C=c)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def svm_linear(x, y, c, oneVone):
    if oneVone is True:
        model = svm.SVC(kernel='linear', C=c)
    else:
        model = svm.LinearSVC(C=c)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime

def svm_poly(x, y, c, deg):
    model = svm.SVC(kernel='poly', degree=deg, C=c)
    start = time.time()
    model.fit(x, y)
    end = time.time()
    trainingTime = end - start
    return model, trainingTime