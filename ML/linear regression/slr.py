import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./50_Startups.csv')
    print(df.describe())

    # Divide data into x's vs y
    X = df.iloc[:,0].values
    Y = df.iloc[:,-1].values

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create LR Model
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Predictions
    Y_pred = regressor.predict(X_test)

    # Plot y_test vs y_pred
    plt.plot(Y_pred, color="blue", label="predictions")
    plt.plot(Y_test, color="red", label="test")
    plt.show()

    # Out of the box predictions
    data = [[80000]]
    new_df = pd.DataFrame(data)
    new_df = sc.transform(new_df)
    single = regressor.predict(new_df)
    print(single)

if __name__ == "__main__":
    main()