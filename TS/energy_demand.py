# ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./TimeSeries_TotalSolarGen_and_Load_IT_2016.csv')

    # make TS
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

    plt.figure(figsize=(14,6))

    plt.plot(df['utc_timestamp'], df['IT_load_new'], label='Load')
    plt.plot(df['utc_timestamp'], df['IT_solar_generation'], label='Solar Generation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Load and Solar Generation over Time')
    plt.show()


    plt.figure(figsize=(14,6))

    #plt.plot(df['utc_timestamp'], df['IT_load_new'], label='Load')
    plt.plot(df['utc_timestamp'], df['IT_solar_generation'], label='Solar Generation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Load and Solar Generation over Time')
    plt.show()

    # EDA
    print(df.isnull().sum())
    df['IT_load_new'].fillna(method='ffill', inplace=True)
    print("Missing values after filling:")
    print(df.isnull().sum())

    # ADF
    def adf_test(time_series):
        print("ADF TEST RESULTS: ")
        dftest = adfuller(time_series, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

    print("\nADF test for 'IT_load_new' after filling missing values:")
    adf_test(df['IT_load_new'])

    print("\nADF test for 'IT_solar_generation':")
    adf_test(df['IT_solar_generation'])

    # plot ACF/PACF
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    plot_acf(df['IT_load_new'], lags=50, zero=False, ax=ax1)
    plot_pacf(df['IT_load_new'], lags=50, zero=False, ax=ax2)
    plt.show()

    # TRAIN TEST SPLIT
    train_size = int(len(df['IT_load_new']) * 0.8)
    train, test = df['IT_load_new'][:train_size], df['IT_load_new'][train_size:]

    # fit ARIMA model 1
    model = ARIMA(train, order=(2,0,2))
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

    # RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    print(rmse)

    # fit ARIMA model 2
    model2 = ARIMA(train, order=(2,1,2))
    model2_fit = model2.fit()

    # make predictions
    predictions2 = model2_fit.predict(start=len(train), end=len(train)+len(test)-1)

    # RMSE
    rmse2 = sqrt(mean_squared_error(test, predictions2))
    print(rmse2)

    # fit ARIMA model 3
    model3 = ARIMA(train, order=(2,2,2))
    model3_fit = model3.fit()

    # make predictions
    predictions3 = model3_fit.predict(start=len(train), end=len(train)+len(test)-1)

    # RMSE
    rmse3 = sqrt(mean_squared_error(test, predictions3))
    print(rmse3)

    plt.figure(figsize=(14,6))
    plt.plot(df['utc_timestamp'][train_size:], test, label='Actual')
    plt.plot(df['utc_timestamp'][train_size:], predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Actual vs Predicted Load Values')
    plt.show()

    #plot ACF/PACF
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    plot_acf(df['IT_solar_generation'], lags=50, zero=False, ax=ax1)
    plot_pacf(df['IT_solar_generation'], lags=50, zero=False, ax=ax2)
    plt.show()

    # TRAIN TEST SPLIT
    train_size = int(len(df['IT_solar_generation']) * 0.8)
    train, test = df['IT_solar_generation'][:train_size], df['IT_solar_generation'][train_size:]

    # fit ARIMA model
    model = ARIMA(train, order=(2,0,2))
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

    # RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    print(rmse)

    # plot
    plt.figure(figsize=(14,6))
    plt.plot(df['utc_timestamp'][train_size:], test, label='Actual')
    plt.plot(df['utc_timestamp'][train_size:], predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Actual vs Predicted Load Values')
    plt.show()


if __name__ == "__main__":
    main()