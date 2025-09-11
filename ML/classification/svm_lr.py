import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # EDA
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.dropna(how='any', inplace=True)

    # X and Y
    X = df.drop(columns=['customerID', 'Churn'], axis=1)
    Y = df.Churn.values

    # Feature Encoding
    X = pd.get_dummies(X, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    # Feature Scaling
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # kNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_sc, Y_train)

    Y_pred_knn = knn_model.predict(X_test_sc)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_sc, Y_train)

    Y_pred_dt = dt_model.predict(X_test_sc)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(X_train_sc, Y_train)

    Y_pred_rf = rf_model.predict(X_test_sc)

    # Naive Bayes
    nb_model = BernoulliNB()
    nb_model.fit(X_train_sc, Y_train)

    Y_pred_nb = nb_model.predict(X_test_sc)

    # SVM
    svm_model = SVC()
    svm_model.fit(X_train_sc, Y_train)

    Y_pred_svm = svm_model.predict(X_test_sc)
    
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_sc, Y_train)

    Y_pred_lr = lr_model.predict(X_test_sc)


    print("knn:", accuracy_score(Y_pred_knn, Y_test))
    print("decision tree:", accuracy_score(Y_pred_dt, Y_test))
    print("random forest:", accuracy_score(Y_pred_rf, Y_test))
    print("naive bayes:", accuracy_score(Y_pred_nb, Y_test))
    print("SVM:", accuracy_score(Y_pred_svm, Y_test))
    print("logistic regression:", accuracy_score(Y_pred_lr, Y_test))

    # print("knn report:", classification_report(Y_test, Y_pred_knn))
    # print("decision tree report:", classification_report(Y_test, Y_pred_dt))
    # print("random forest report:", classification_report(Y_test, Y_pred_rf))
    # print("naive bayes report:", classification_report(Y_test, Y_pred_nb))
    # print("SVM:", classification_report(Y_test, Y_pred_svm))
    # print("logistic regression:", classification_report(Y_test, Y_pred_lr))

if __name__ == "__main__":
    main()