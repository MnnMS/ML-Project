from pre_processing import *
import multivariable_model
import Normal_model
import sklearn.metrics as sm

data = pd.read_csv('AppleStore_training.csv')

data.dropna(how='any', inplace=True)
Y=data['user_rating']

# model 1 multivariable
#X, cols = multivariable_model.pre_process(data)

# model 2 Normal
X, cols = Normal_model.pre_process(data)

X = One_Hot_Encoding(X,cols)
X = featureScaling(np.array(X),0,1)

# model 1
#prediction, y_test = multivariable_model.fit(X,Y)
# model 2
prediction = Normal_model.fit(X,Y)

# multi
#test = y_test
# normal
test = Y

print('Mean Square Error', metrics.mean_squared_error(np.asarray(test), prediction))
print("Mean squared error =", round(sm.mean_squared_error(np.asarray(test), prediction), 2))
print("Median absolute error =", round(sm.median_absolute_error(np.asarray(test), prediction), 2))
print("Explain variance score =", round(sm.explained_variance_score(np.asarray(test), prediction), 2))
print("R2 score =", round(sm.r2_score(np.asarray(test), prediction), 2))