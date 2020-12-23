from pre_processing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv('AppleStore_training.csv')

data.dropna(how='any',inplace=True)
data.drop(columns=['currency', 'track_name'], inplace=True)
# all_data = data.iloc[:,:]
X=data.iloc[:,0:15]
Y=data['user_rating']

cols=['prime_genre','ver','cont_rating']
X = One_Hot_Encoding(X,cols)
X = featureScaling(np.array(X),0,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, shuffle=True)

# corr = all_data.corr()
# top_feature = corr.index[abs(corr['user_rating']>0.3)]
# top_corr = all_data[top_feature].corr()

cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction= cls.predict(X_test)

print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_rate_value=np.asarray(y_test)[5]
predicted_rate_value=prediction[5]

print('True rate value s is : ' + str(true_rate_value))
print('Predicted rate value  is : ' + str(predicted_rate_value))



