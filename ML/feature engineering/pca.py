import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

    accuracy = accuracy_score(Y_test, y_pred_orig)
    print("Accuracy:", accuracy)

    # PCA
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train, Y_train)
    X_test_pca = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    # PCA model
    model_pca = LogisticRegression()
    model_pca.fit(X_train_pca, Y_train)
    y_pred_pca = model_pca.predict(X_test_pca)

    accuracy_pca = accuracy_score(Y_test, y_pred_pca)
    print("Accuracy:", accuracy_pca)

if __name__ == "__main__":
    main()