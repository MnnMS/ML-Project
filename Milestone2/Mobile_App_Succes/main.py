import pickle

from classification_models import *
from regression_models import *
from pre_processing import *
import pandas as pd
import sklearn.metrics as sm

type = input('Enter 1 for Regression or 2 For Classification: ')

if type == '1':
    train = input('Enter 1 to train the models 2 to test it: ')
    flag = True if train == '2' else False
    if not flag:
        data = pd.read_csv('AppleStore_training.csv')
    else:
        filename = input('Enter test file name: ')
        data = pd.read_csv(filename)

    # preprocessing
    #corr_matrix(data, False)
    x_train, x_test, y_train, y_test = pre_process(data,reg=True,test=flag)

    acc = {}
    # train
    if not flag:
        multi_model, multi_time = multivariable_reg(x_train, y_train)
        pred = multi_model.predict(x_test)
        MSE = round(sm.mean_squared_error(np.asarray(y_test), pred), 2)
        R2score = round(sm.r2_score(np.asarray(y_test), pred), 2)
        accuracy = [MSE, R2score]
        acc['multivariable'] = accuracy
        filename = 'multivariable_model.sav'
        pickle.dump(multi_model, open(filename, 'wb'))

        poly_model, poly_time = polynomial_reg(x_train, y_train)
        poly_features = PolynomialFeatures(degree=3)
        X_test = poly_features.fit_transform(x_test)
        pred = poly_model.predict(X_test)
        MSE = round(sm.mean_squared_error(np.asarray(y_test), pred), 2)
        R2score = round(sm.r2_score(np.asarray(y_test), pred), 2)
        accuracy = [MSE, R2score]
        acc['polynomial'] = accuracy
        filename = 'polynomial_model.sav'
        pickle.dump(poly_model, open(filename, 'wb'))

    else:
        files = ['multivariable_model.sav','polynomial_model.sav']
        for file in files:
            model = pickle.load(open(file, 'rb'))
            if file == 'polynomial_model.sav':
                poly_features = PolynomialFeatures(degree=3)
                x_test = poly_features.fit_transform(x_test)
            pred = model.predict(x_test)
            MSE = round(sm.mean_squared_error(np.asarray(y_test), pred), 2)
            R2score = round(sm.r2_score(np.asarray(y_test), pred), 2)
            accuracy = [MSE, R2score]
            acc[file.split('.')[0]] = accuracy

    for modelName in acc:
        print('Model Name: ' + modelName)
        scores = [i for i in acc[modelName]]
        print('MSE = {}\nR2score = {}\n'.format(scores[0], scores[1]))

elif type == '2':
    train = input('Enter 1 to train the models 2 to test it: ')
    flag = True if train == '2' else False
    if not flag:
        data = pd.read_csv('AppleStore_training_classification.csv')
    else:
        filename = input('Enter test file name: ')
        data = pd.read_csv(filename)

    # preprocessing
    #corr_matrix(data, False)
    x_train, x_test, y_train, y_test = pre_process(data,reg=False,test=flag)

    acc = {}
    #train
    if not flag:
        decTree_model, decTree_time = decision_tree(x_train,y_train)
        pred = decTree_model.predict(x_test)
        accuracy = np.mean(pred == y_test)
        acc['decision_tree'] = round(accuracy*100)
        filename = 'decTree_model.sav'
        pickle.dump(decTree_model, open(filename, 'wb'))

        adboost_model, adboost_time = adaboost(x_train, y_train)
        pred = adboost_model.predict(x_test)
        accuracy = np.mean(pred == y_test)
        acc['Adaboost'] = round(accuracy * 100)
        filename = 'adaboost_model.sav'
        pickle.dump(adboost_model, open(filename, 'wb'))

        # logReg_model, logReg_time = logistic_reg(x_train, y_train)
        # pred = logReg_model.predict(x_test)
        # accuracy = np.mean(pred == y_test)
        # acc['logistic_regression'] = round(accuracy*100)
        # filename = 'logReg_model.sav'
        # pickle.dump(logReg_model, open(filename, 'wb'))

        svmlinear_model, svmlinear_time = svm_linear(x_train, y_train, c=1, oneVone=True)
        pred = svmlinear_model.predict(x_test)
        accuracy = np.mean(pred == y_test)
        acc['svm_linear_1v1'] = round(accuracy * 100)
        filename = 'svm_linear_1v1_model.sav'
        pickle.dump(svmlinear_model, open(filename, 'wb'))

        knn_model, knn_time = KNN(x_train, y_train)
        pred = knn_model.predict(x_test)
        accuracy = np.mean(pred == y_test)
        acc['KNN'] = round(accuracy * 100)
        filename = 'knn_model.sav'
        pickle.dump(knn_model, open(filename, 'wb'))

    else:
        files = ['decTree_model.sav','adaboost_model.sav','svm_linear_1v1_model.sav', 'knn_model.sav']
        for file in files:
            model = pickle.load(open(file, 'rb'))
            pred = model.predict(x_test)
            accuracy = np.mean(pred == y_test)
            acc[file.split('.')[0]] = round(accuracy * 100)

    for modelName in acc:
        print('Model Name: {}, Model Accuracy: {}%\n'.format(modelName, acc[modelName]))

else:
    print('Invalid')