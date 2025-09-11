import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def main():
    X, Y = make_regression(n_samples=10000, n_features=10, n_informative=3)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # DECISION TREE
    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(X_train, Y_train)
    Y_pred_dt = model_dt.predict(X_test)

    # BAGGING
    bag = BaggingRegressor(estimator=DecisionTreeRegressor(),
                            n_estimators=500,
                            max_samples=0.5,
                            bootstrap=True,
                            random_state=42)
    bag.fit(X_train, Y_train)
    Y_pred_bag = bag.predict(X_test)

    # RANDOM FOREST
    model_rf = RandomForestRegressor(random_state=42, n_estimators=500)
    model_rf.fit(X_train, Y_train)
    Y_pred_rf = model_rf.predict(X_test)

    # BAGGING USING SVM
    bag_svm = BaggingRegressor(estimator=SVR(),
                            n_estimators=500,
                            max_samples=0.25,
                            bootstrap=True,
                            random_state=42)
    bag_svm.fit(X_train, Y_train)
    Y_pred_bag_svm = bag_svm.predict(X_test)

    # PASTING
    pasting = BaggingRegressor(estimator=DecisionTreeRegressor(),
                            n_estimators=500,
                            max_samples=0.5,
                            bootstrap=False,
                            random_state=42)
    pasting.fit(X_train, Y_train)
    Y_pred_pasting = pasting.predict(X_test)

    print("Decision tree:", r2_score(Y_test, Y_pred_dt))
    print("Bagging using DT:", r2_score(Y_test, Y_pred_bag))
    print("Random forest:", r2_score(Y_test, Y_pred_rf))
    print("Bagging using SVM:", r2_score(Y_test, Y_pred_bag_svm))
    print("Pasting:", r2_score(Y_test, Y_pred_pasting))

if __name__ == "__main__":
    main()