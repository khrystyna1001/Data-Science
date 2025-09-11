import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def main():
    X, Y = make_classification(n_samples=10000, n_features=10, n_informative=3)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # DECISION TREE
    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(X_train, Y_train)
    Y_pred_dt = model_dt.predict(X_test)

    # BAGGING
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(),
                            n_estimators=500,
                            max_samples=0.5,
                            bootstrap=True,
                            random_state=42)
    bag.fit(X_train, Y_train)
    Y_pred_bag = bag.predict(X_test)

    # RANDOM FOREST
    model_rf = RandomForestClassifier(random_state=42, n_estimators=500)
    model_rf.fit(X_train, Y_train)
    Y_pred_rf = model_rf.predict(X_test)

    # BAGGING USING SVM
    bag_svm = BaggingClassifier(estimator=SVC(),
                            n_estimators=500,
                            max_samples=0.25,
                            bootstrap=True,
                            random_state=42)
    bag_svm.fit(X_train, Y_train)
    Y_pred_bag_svm = bag_svm.predict(X_test)

    # PASTING
    pasting = BaggingClassifier(estimator=DecisionTreeClassifier(),
                            n_estimators=500,
                            max_samples=0.5,
                            bootstrap=False,
                            random_state=42)
    pasting.fit(X_train, Y_train)
    Y_pred_pasting = pasting.predict(X_test)

    print("Decision tree:", accuracy_score(Y_test, Y_pred_dt))
    print("Bagging using DT:", accuracy_score(Y_test, Y_pred_bag))
    print("Random forest:", accuracy_score(Y_test, Y_pred_rf))
    print("Bagging using SVM:", accuracy_score(Y_test, Y_pred_bag_svm))
    print("Pasting:", accuracy_score(Y_test, Y_pred_pasting))

if __name__ == "__main__":
    main()