import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pre_processing import *


dataset = pd.read_csv('AppleStore_training_classification.csv')
dataset.dropna(how='any', inplace=True)
#X = dataset.iloc[:, :-1].values
Y = []
y = dataset['rate']
for val in y:
    if val == 'Low':
        Y.append(0)
    elif val == 'High':
        Y.append(2)
    else:
        Y.append(1)
def pre_process(data):
    #data.drop(columns=['currency', 'ver', 'id','track_name'], inplace=True)
    # all_data = data.iloc[:,:]
    X = data.iloc[:, [2,4,5,6,10,11,12]]
    cols = ['prime_genre']
    return X, cols

X,cols = pre_process(dataset)
# print(X)


X = One_Hot_Encoding(X, cols)
#X = featureScaling(np.array(X), 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
# knn = KNeighborsClassifier(n_neighbors=40)
# knn.fit(X_train, y_train)
# pred_i = knn.predict(X_test)
# print('Accuracy: ',np.mean(pred_i == y_test))

