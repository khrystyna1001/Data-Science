import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df.dropna(how='any', inplace=True)

    X = df.drop(['customerID', 'Churn'], axis=1)
    Y = df.Churn.values

    # FEATURE ENCODING
    X = pd.get_dummies(X, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    
    # FEATURE SCALING
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_sc, y_train)
    knn_pred = knn_model.predict(X_test_sc)

    # DECISION TREE
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train_sc, y_train)
    dt_pred = model_dt.predict(X_test_sc)

    # RANDOM FOREST
    model_rf = RandomForestClassifier(n_estimators=200)
    model_rf.fit(X_train_sc, y_train)
    rf_pred = model_rf.predict(X_test_sc)

    # NAIVE BAYES
    model_nb = BernoulliNB()
    model_nb.fit(X_train_sc, y_train)
    nb_pred = model_nb.predict(X_test_sc)

    # SUPPORT VECTOR MACHINE
    model_svc = SVC()
    model_svc.fit(X_train_sc, y_train)
    svc_pred = model_svc.predict(X_test_sc)

    # LOGISTIC REGRESSION
    model_lr = LogisticRegression()
    model_lr.fit(X_train_sc, y_train)
    lr_pred = model_lr.predict(X_test_sc)

    print("============ ACCURACY SCORE ===============")
    print("KNN", accuracy_score(y_test, knn_pred)*100)
    print("DT", accuracy_score(y_test,dt_pred)*100)
    print("RF", accuracy_score(y_test,rf_pred)*100)
    print("NB", accuracy_score(y_test,nb_pred)*100)
    print("SVM", accuracy_score(y_test,svc_pred)*100)
    print("LR", accuracy_score(y_test,lr_pred)*100)

    # ENSEMBLE LEARNING

    # ADA BOOST
    ada_model = AdaBoostClassifier(n_estimators=200)
    ada_model.fit(X_train_sc, y_train)
    ada_pred = ada_model.predict(X_test_sc)

    # GRADIENT BOOST
    gradient_model = GradientBoostingClassifier(n_estimators=200)
    gradient_model.fit(X_train_sc, y_train)
    gb_pred = gradient_model.predict(X_test_sc)

    print("============ ENSEMBLE LEARNING ===============")
    print("ADA BOOST", accuracy_score(y_test, ada_pred)*100)
    print("GRADIENT BOOST", accuracy_score(y_test, gb_pred)*100)

    # XGB
    Y[Y=='No'] = 0
    Y[Y=='Yes'] = 1
    Y = Y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    xgb_model = XGBClassifier(n_estimators=200, max_depth=4)
    xgb_model.fit(X_train_sc, y_train)
    xgb_pred = xgb_model.predict(X_test_sc)

    print("XTREME GRADIENT BOOST", accuracy_score(y_test, xgb_pred)*100)
    
if __name__ == "__main__":
    main()