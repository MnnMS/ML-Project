from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer
nltk.download('stopwords')


def semantic(X):
    X = X.apply(lambda x: " ".join(x.lower() for x in x.split()))
    X = X.str.replace('[^\w\s]', '')
    stop = stopwords.words('english')
    X = X.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    st = PorterStemmer()
    X = X.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    for i in range(len(X)):
        X[i] = TextBlob(X[i]).sentiment.polarity
    return X


