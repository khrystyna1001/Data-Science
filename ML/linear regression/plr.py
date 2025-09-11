import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

def main():
    pd.set_option('display.max_column', None)
    X = 6 * np.random.rand(200,1)-3
    Y = 0.8 * X**2 + 0.9 * X + 2 + np.random.rand(200,1)

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # LR model
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    # Prediction
    Y_Pred = lr.predict(X_test)
    r2_score_lr = r2_score(Y_test, Y_Pred)
    print("R2 score", r2_score_lr)

    # Plot
    plt.plot(X_train, lr.predict(X_train), color="r")
    plt.plot(X, Y, "b.")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # POLYNOMIAL LINEAR REGRESSION
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_train_trans = poly.fit_transform(X_train)
    X_test_trans = poly.transform(X_test)

    lr2 = LinearRegression()
    lr2.fit(X_train_trans, Y_train)

    y_pred = lr2.predict(X_test_trans)
    print("R2 SCORE", r2_score(Y_test, y_pred))

    print("Coefficients", lr.coef_)
    print("Intercept", lr.intercept_)

    x_new = np.linspace(-3, 3, 200).reshape(200,1)
    x_new_poly = poly.transform(x_new)
    y_new = lr2.predict(x_new_poly)

    plt.plot(x_new, y_new, "r-", linewidth=2, label="Prediction")
    plt.plot(X_train, Y_train, "b.", label="Training")
    plt.plot(X_test, Y_test, "g.", label="Testing")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    def polynomial_regression(degree):
        X_new=np.linspace(-3, 3, 100).reshape(100, 1)
        X_new_poly = poly.transform(X_new)

        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
                ("poly_features", polybig_features),
                ("std_scaler", std_scaler),
                ("lin_reg", lin_reg),
            ])
        polynomial_regression.fit(X, Y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig,'r', label="Degree " + str(degree), linewidth=2)

        plt.plot(X_train, Y_train, "b.", linewidth=3)
        plt.plot(X_test, Y_test, "g.", linewidth=3)
        plt.legend(loc="upper left")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.axis([-3, 3, 0, 10])
        plt.show()

    polynomial_regression(200)


if __name__ == "__main__":
    main()