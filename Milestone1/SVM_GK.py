from pre_processing import *
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

def pre_process(data):
    # Cleaning content rating
    RatingL = data['cont_rating'].unique()
    RatingDict = {}
    for i in range(len(RatingL)):
        RatingDict[RatingL[i]] = i
    data['cont_rating'] = data['cont_rating'].map(RatingDict).astype(int)

    # Cleaning prices
    def price_clean(price):
        if price == '0':
            return 0
        else:
            price = float(price)
            return price

    data['price'] = data['price'].map(price_clean).astype(float)
    data['rating_count_tot'] = data['rating_count_tot'].astype(int)
    data['rating_count_ver'] = data['rating_count_ver'].astype(int)
    data['ipadSc_urls.num'] = data['ipadSc_urls.num'].astype(int)
    data['lang.num'] = data['lang.num'].astype(int)

    X = data.iloc[:, [5, 6, 10, 11, 12, 13]]
    #X = data.iloc[:, 1:14]
    Y = []
    target = data['rate']
    for val in target:
        if val == 'Low':
            Y.append(0)
        elif val == 'High':
            Y.append(2)
        else:
            Y.append(1)
    cols = ['prime_genre']
    return X,Y,cols

def fit(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=0,shuffle=True)

    C = 1
    #svc = svm.SVC(kernel='rbf', C=C).fit(X_train, y_train)
    #svc_ovr = OneVsRestClassifier(svm.SVC(kernel='rbf', C=C)).fit(X_train, y_train)
    #svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    #svc = svm.LinearSVC(C=C).fit(X_train, y_train)
    #svc = svm.SVC(kernel='poly', degree=5, C=C).fit(X_train, y_train)
    # accuracy = svc_ovr.score(X_test, y_test)
    # print('One VS Rest SVM accuracy: ' + str(accuracy))
    prediction= svc.predict(X_test)

    return prediction, y_test

data = pd.read_csv('AppleStore_training_classification.csv')
data.dropna(how='any', inplace=True)

X, Y, cols = pre_process(data)
X = One_Hot_Encoding(X, cols)
X = featureScaling(np.array(X), 0, 1)
pred, ytest = fit(X, Y)
accuracy = np.mean(pred == ytest)
print(accuracy)