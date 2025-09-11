import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    pd.set_option('display.max_column', None)

    df = pd.read_csv('./50_Startups.csv')
    # print(df.info())

    X = df.iloc[:,0].values
    Y = df.iloc[:,-1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print(len(X_test))
    print(len(X_train))

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # LR model
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # Plot
    plt.plot(Y_test, color="blue", label="test")
    plt.plot(y_pred, color="red", label="predictions")
    plt.show()

    # Metrics
    mae = mean_absolute_error(y_pred, Y_test)
    mse = mean_squared_error(y_pred, Y_test)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_pred, Y_test)
    print(r2)

    # Calculate the Adjusted R squared
    n = X_test.shape[0]
    k = X_test.shape[1]

    adjusted_r2 = 1 - (1-r2)*(n-1)/(n-1-k)
    print(adjusted_r2)

if __name__ == "__main__":
    main()