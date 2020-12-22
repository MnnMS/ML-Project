from pre_processing import *
import pandas as pd
import numpy as np

data = pd.read_csv('AppleStore_training.csv')
data.dropna(how='any',inplace=True)
X=data.iloc[:,0:15]
Y=data['user_rating']
cols=['prime_genre','currency','track_name','ver','cont_rating']
X = One_Hot_Encoding(X,cols)
X = featureScaling(np.array(X),0,1);
