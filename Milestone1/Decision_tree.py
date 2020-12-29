from pre_processing import *
from sklearn import tree
from sklearn.model_selection import train_test_split

def pre_process(data):
    X = data.iloc[:,[4,5,6,9,10,11,12,13]]
    Y = data.iloc[:,-1]
    col = ['prime_genre','cont_rating']
    return X, Y, col

def fit(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    return prediction, y_test


data = pd.read_csv('AppleStore_training_classification.csv')
data.dropna(how='any', inplace=True)
X, Y, cols = pre_process(data)
X = One_Hot_Encoding(X, cols)
X = featureScaling(np.array(X), 0, 1)
pred, y_test = fit(X, Y)
accuracy = np.mean(pred == y_test)*100
print(accuracy)
