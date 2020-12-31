import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from numpy import nan

def corr_matrix(data, reg):
    corr = data.corr()
    y_col_name = 'user_rating' if reg else 'rate'
    top_features = corr.index[corr[y_col_name] >= 0.05]
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

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

def pre_process(data, reg, test):
    data.drop(columns=['id', 'track_name', 'currency', 'vpp_lic', 'ver'],  inplace=True)

    data[['prime_genre']] = data[['prime_genre']].replace('0', nan)
    if test:
        si_mean = SimpleImputer(strategy='mean', add_indicator=False)
        si_mode = SimpleImputer(strategy='most_frequent', add_indicator=False)
        si_const = SimpleImputer(strategy='constant', add_indicator=False)
        data[['size_bytes']] = si_mode.fit_transform(data[['size_bytes']])
        data[['rating_count_tot']] = si_mode.fit_transform(data[['rating_count_tot']])
        data[['rating_count_ver']] = si_mode.fit_transform(data[['rating_count_ver']])
        data[['sup_devices.num']] = si_mean.fit_transform(data[['sup_devices.num']])
        data[['ipadSc_urls.num']] = si_mean.fit_transform(data[['ipadSc_urls.num']])
        data[['lang.num']] = si_mean.fit_transform(data[['lang.num']])
        data[['price']] = si_mode.fit_transform(data[['price']])
        data[['cont_rating']] = si_mode.fit_transform(data[['cont_rating']])
        data[['prime_genre']] = si_const.fit_transform(data[['prime_genre']])
        if reg:
            data[['user_rating_ver']] = si_mode.fit_transform(data[['user_rating_ver']])
            data[['user_rating']] = si_mode.fit_transform(data[['user_rating']])
        else:
            data[['rate']] = si_mode.fit_transform(data[['rate']])
    else:
        data.dropna(how='any', inplace=True)

    #datatypes
    data['price'] = data['price'].astype(float)
    data['size_bytes'] = data['size_bytes'].astype(float)
    data['rating_count_tot'] = data['rating_count_tot'].astype(int)
    data['rating_count_ver'] = data['rating_count_ver'].astype(int)
    data['sup_devices.num'] = data['sup_devices.num'].astype(int)
    data['ipadSc_urls.num'] = data['ipadSc_urls.num'].astype(int)
    data['lang.num'] = data['lang.num'].astype(int)
    if reg:
        data['user_rating'] = data['user_rating'].astype(float)
    else:
        Y = []
        target = data['rate']
        for val in target:
            if val == 'Low':
                Y.append(0)
            elif val == 'High':
                Y.append(2)
            else:
                Y.append(1)
        data['rate'] = Y
    #data.info()

    #split data
    X = data.iloc[:,:-1]
    Y = data.iloc[:, -1]

    #encode categorical features
    #le_contRating = LabelEncoder()
    #X['cont_rating'] = le_contRating.fit_transform(X['cont_rating'])

    X = One_Hot_Encoding(X, ['prime_genre', 'cont_rating'])
    if test:
        X.drop(columns=['missing_value'], inplace=True)
    X = featureScaling(np.array(X), 0, 1)
    if not test:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        return x_train, x_test, y_train, y_test   # try random_state=0,shuffle=True
    else:
        return [],X,[],Y

    #feature scaling
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)




# data.info() // data types
#
# correlation
#
# feature selection
#
# missing values (mean, mode)
#
# one hot
#
# split data
#
# feature scaling sklearn