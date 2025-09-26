import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./spam.csv')

    df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)
    
    # Create bag of words representation using CountVectorizer
    v = CountVectorizer()
    X_train_cv = v.fit_transform(X_train.values)
    X_test_cv = v.transform(X_test.values)
    X_train_cv.toarray()[:2][0]

    print(v.vocabulary_)

    X_train_np = X_train_cv.toarray()
    print(np.where(X_train_np[0]!=0))

    # Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train_cv, y_train)
    y_pred = model.predict(X_test_cv)
    print(classification_report(y_test, y_pred))

    # Test on a random data point
    message = {"Nothing but hits from you"}
    message_cnt = v.transform(message)
    print(model.predict(message_cnt))

if __name__ == "__main__":
    main()