import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print('Accuracy:', accuracy)
    return accuracy

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

    # RANDOM HPO
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=110, num=11)]
    min_samples_leaf = [1,2,4,10,20,50,100]
    min_samples_split = [2,3,4,5,8,10,20,50,100,200]
    bootstrap = [True, False]

    random_grid = { 'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'bootstrap': bootstrap,
                    'min_samples_leaf': min_samples_leaf,
                   }
    

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, n_jobs=-1)
    rf_random.fit(X_train, Y_train)

    base_model = RandomForestClassifier(n_estimators=5, random_state=42)
    base_model.fit(X_train, Y_train)
    base_accuracy = evaluate(base_model, X_test, Y_test)

    best_random = rf_random.best_estimator_
    print(best_random)

if __name__ == "__main__":
    main()