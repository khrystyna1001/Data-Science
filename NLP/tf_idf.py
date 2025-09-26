import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./Ecommerce_data.csv')

    df['label_num'] = df['label'].map({
        'Household': 1,
        'Electronics': 2,
        'Clothing & Accessories': 3,
        'Books': 4
    })
    
    print(df.head(5))
    
    X_train, X_test, y_train, y_test = train_test_split(df.Text, df.label_num, test_size=0.2)

    # Vectorizer
    v = TfidfVectorizer()
    X_train_v = v.fit_transform(X_train)
    X_test_v = v.transform(X_test)
    # print(v.vocabulary_)

    # Classification Model
    model = DecisionTreeClassifier()
    model.fit(X_train_v,y_train)

    y_pred = model.predict(X_test_v)
    print(classification_report(y_test, y_pred))

    # Testing on a data point
    msg = ["Designer Women's Art Mysore Silk Saree"]
    msg_v = v.transform(msg)
    print(model.predict(msg_v))

if __name__ == "__main__":
    main()