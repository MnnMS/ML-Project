from pre_processing import *
import multivariable_model
import polynomial_model
import sklearn.metrics as sm
from bonus import semantic

data = pd.read_csv('AppleStore_training.csv')

data.dropna(how='any', inplace=True)
Y=data['user_rating']

# remove/add comments to use specific model

# model 1 multivariable
X, cols = multivariable_model.pre_process(data)



# model 2 Polynomial
#X,cols = polynomial_model.pre_process(data)



#**BONUS**
# data2 = pd.read_csv('AppleStore_u_description(Optional_Bonus).csv')
# data2.dropna(how='any', inplace=True)
# B = data2['app_desc']
# newB = semantic(B)
# X['desc'] =  newB


X = One_Hot_Encoding(X, cols)
X = featureScaling(np.array(X), 0, 1)


# model 1
prediction, y_test = multivariable_model.fit(X,Y)


#model 2
#prediction,y_test = polynomial_model.train(X,Y)

# multi
test = y_test

#Polynomial
#test = y_test

print('Mean Square Error', metrics.mean_squared_error(np.asarray(test), prediction))
print("Mean squared error =", round(sm.mean_squared_error(np.asarray(test), prediction), 2))
print("Median absolute error =", round(sm.median_absolute_error(np.asarray(test), prediction), 2))
print("Explain variance score =", round(sm.explained_variance_score(np.asarray(test), prediction), 2))
print("R2 score =", round(sm.r2_score(np.asarray(test), prediction), 2))