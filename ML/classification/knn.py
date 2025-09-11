import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    
    # FEATURE SCALING
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # INITIATE KNN CLASSIFIER
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_sc, Y_train)

    # PREDICTION
    Y_pred = knn_model.predict(X_test_sc)
    print(accuracy_score(Y_test, Y_pred)*100)

    print(X_test.columns)

    # NEW DATA PREDICTION
    data = [[0, 2, 87, 178, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    
    data_sc = sc.transform(data)
    single = knn_model.predict(data_sc)
    print(single)
    
if __name__ == "__main__":
    main()