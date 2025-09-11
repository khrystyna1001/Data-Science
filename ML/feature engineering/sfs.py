import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
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

    # SFS

    model = LogisticRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=5)
    sfs = sfs.fit(X_train, Y_train)

    selected_features = X.columns[sfs.get_support()]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model_sfs = LogisticRegression()
    model_sfs.fit(X_train_selected, Y_train)
    y_pred_sfs = model_sfs.predict(X_test_selected)

    accuracy = accuracy_score(Y_test, y_pred_sfs)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()