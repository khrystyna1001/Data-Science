import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./boston.csv')
    
    X = df.drop('Price', axis=1)
    Y = df['Price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    Y_pred = lr.predict(X_test)
    print(r2_score(Y_test, Y_pred))

    # RIDGE REGRESSION
    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(X_train, Y_train)
    Y_pred_ridge = ridge_model.predict(X_test)
    print(r2_score(Y_test, Y_pred_ridge))

    # LASSO REGRESSION
    lasso_model = Lasso(alpha=1.0)
    lasso_model.fit(X_train, Y_train)
    Y_pred_lasso = lasso_model.predict(X_test)
    print(r2_score(Y_test, Y_pred_lasso))


    # Identify coefficients with bad slope results
    bad_features = np.where(lasso_model.coef_==0)[0]

    X_train_filtered = X_train.drop(X_train.columns[bad_features], axis=1)
    X_test_filtered = X_test.drop(X_test.columns[bad_features], axis=1)

    lr_model_filtered = LinearRegression()
    lr_model_filtered.fit(X_train_filtered, Y_train)

    lasso_model_filtered = Lasso(alpha=0.1)
    lasso_model_filtered.fit(X_train_filtered, Y_train)

    Y_pred_lr_filtered = lr_model_filtered.predict(X_test_filtered)
    print(r2_score(Y_pred_lr_filtered, Y_test))

    Y_pred_lasso_filtered = lasso_model_filtered.predict(X_test_filtered)
    print(r2_score(Y_pred_lasso_filtered, Y_test))

    # Coefficients close to 0
    small_features = np.where((lasso_model.coef_<0.05) & (lasso_model.coef_>-0.05))

    X_train_new = X_train.drop(X_train.columns[small_features], axis=1)
    X_test_new = X_test.drop(X_test.columns[small_features], axis=1)

    lasso_filtered_small = Lasso(alpha=0.1)
    lasso_filtered_small.fit(X_train_new, Y_train)
    
    Y_pred_new = lasso_filtered_small.predict(X_test_new)
    print(r2_score(Y_pred_new, Y_test))

    
if __name__ == "__main__":
    main()