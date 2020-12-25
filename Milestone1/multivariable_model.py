from pre_processing import *

def pre_process(data):
    #data.drop(columns=['currency','ver','id'], inplace=True)
    # all_data = data.iloc[:,:]
    #cols=['prime_genre','cont_rating','track_name', 'vpp_lic']

    # Cleaning content rating
    RatingL = data['cont_rating'].unique()
    RatingDict = {}
    for i in range(len(RatingL)):
        RatingDict[RatingL[i]] = i
    data['cont_rating'] = data['cont_rating'].map(RatingDict).astype(int)

    # # Cleaning prices
    # def price_clean(price):
    #     if price == '0':
    #         return 0
    #     else:
    #         price = float(price)
    #         return price

    #data['price'] = data['price'].map(price_clean).astype(float)
    data['rating_count_tot'] = data['rating_count_tot'].astype(int)
    data['rating_count_ver'] = data['rating_count_ver'].astype(int)
    data['ipadSc_urls.num'] = data['ipadSc_urls.num'].astype(int)
    data['lang.num'] = data['lang.num'].astype(int)

    X = data.iloc[:, [5, 6, 8, 10, 11, 13, 14]]

    cols = ['prime_genre']
    return X,cols

def fit(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, shuffle=True)

    # corr = all_data.corr()
    # top_feature = corr.index[abs(corr['user_rating']>0.3)]
    # top_corr = all_data[top_feature].corr()

    #cls = linear_model.LinearRegression()
    cls = linear_model.Ridge()
    cls.fit(X_train,y_train)
    prediction= cls.predict(X_test)

    return prediction, y_test