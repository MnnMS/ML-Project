from pre_processing import *

def pre_process(data):
    data.drop(columns=['currency','ver' , 'id'], inplace=True)
    # all_data = data.iloc[:,:]
    X=data.iloc[:,0:12]

    cols=['prime_genre','cont_rating','track_name']
    return X,cols

def fit(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, shuffle=True)

    # corr = all_data.corr()
    # top_feature = corr.index[abs(corr['user_rating']>0.3)]
    # top_corr = all_data[top_feature].corr()

    cls = linear_model.LinearRegression()
    cls.fit(X_train,y_train)
    prediction= cls.predict(X_test)

    return prediction, y_test