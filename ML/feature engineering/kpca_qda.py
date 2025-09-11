import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
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

    # kPCA
    kpca = KernelPCA(n_components=10)
    X_train_kpca = kpca.fit_transform(X_train, Y_train)
    X_test_kpca = kpca.transform(X_test)

    # kPCA model
    model_kpca = LogisticRegression()
    model_kpca.fit(X_train_kpca, Y_train)
    y_pred_kpca = model_kpca.predict(X_test_kpca)

    accuracy_kpca = accuracy_score(Y_test, y_pred_kpca)
    print("Accuracy:", accuracy_kpca)

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    X_train_qda = qda.fit(X_train, Y_train)
    
    y_pred_qda = qda.predict(X_test)
    accuracy_qda = accuracy_score(Y_test, y_pred_qda)
    print("Accuracy:", accuracy_qda)

if __name__ == "__main__":
    main()