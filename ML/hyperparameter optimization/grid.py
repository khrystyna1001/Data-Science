import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from randomm import evaluate

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./breast_cancer.csv')
    
    # EDA
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    Y = df.diagnosis.values

    # TTS
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # FEATURE SCALING
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # MODEL BUILDING
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    result = accuracy_score(Y_test, y_pred)
    print("Accuracy RF:", result)
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix RF:", cm)
    cr = classification_report(Y_test, y_pred)
    print("Classification Report RF:", cr)

    # GRID HPO
    param_grid = {
        'bootstrap': [True],
        'max_depth': [75, 80, 85, 90, 95, 100],
        'max_features': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [200, 250, 270, 300, 350, 400, 450, 500]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    grid_search.best_params_

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, Y_test)

if __name__ == "__main__":
    main()