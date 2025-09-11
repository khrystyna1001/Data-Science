import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') 
    df.dropna(how='any', inplace=True)
    df.drop(['customerID'], axis='columns', inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(['Churn_Yes'], axis=1)
    Y = df['Churn_Yes']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Build model
    model_orig = LogisticRegression()
    model_orig.fit(X_train, Y_train)

    y_pred_orig = model_orig.predict(X_test)

    accuracyy = accuracy_score(Y_test, y_pred_orig)
    print("Accuracy:", accuracyy)

    # CHI SQUARE
    chi2_selector = SelectKBest(chi2, k=5)
    X_train_chi2 = chi2_selector.fit_transform(X_train, Y_train)
    X_test_chi2 = chi2_selector.transform(X_test)

    model_chi2 = LogisticRegression()
    model_chi2.fit(X_train_chi2, Y_train)

    y_pred_chi2 = model_chi2.predict(X_test_chi2)

    accuracy = accuracy_score(Y_test, y_pred_chi2)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()