import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Import Dataset
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./Churn_Modelling.csv')

    X = df.iloc[:, 3:13].values
    y = df.iloc[:, 13].values

    # Feature Encoding
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    print(X)

    # Impute missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Apply to all columns that might have NaNs
    X[:, :] = imputer.fit_transform(X[:, :])

    # TTS
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Classical ML
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred1 = clf.predict(X_test)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm1 = confusion_matrix(y_test, pred1)
    score1 = accuracy_score(y_test, pred1)
    print(cm1)
    print(score1*100)

    # Classical ML
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    pred2 = rfc.predict(X_test)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm2 = confusion_matrix(y_test, pred2)
    score2 = accuracy_score(y_test, pred2)
    print(cm2)
    print(score2*100)

    # ANN
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialize ANN
    classifier = Sequential()
    # I/p layer & first hidden layer (6 neurons)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    # Second hidden layer (6 neurons)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # Third hidden layer (6 neurons)
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # Output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compile ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    y_train[0]
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

    # Predicting the Test set results
    y_pred_ann = classifier.predict(X_test)
    y_pred_ann = (y_pred_ann > 0.5)

    cm3 = confusion_matrix(y_test, y_pred_ann)
    score3 = accuracy_score(y_test, y_pred_ann)
    print(cm3)
    print(score3)

    # ANN HPO

    from scikeras.wrappers import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dense


    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier


    classifier = KerasClassifier(model = build_classifier, verbose=0)

    parameters = {'batch_size': [10, 20],
                'epochs': [5, 6]}

                # Total combinations: 4

    grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', n_jobs = -1, cv = 5)
    grid_search = grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)



if __name__ == "__main__":
    main()