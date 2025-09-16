# UNIVARIATE FACEBOOK PROPHET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./Tesla.csv')

    # EDA
    missing_values = df.isnull().sum()
    statistics = df.describe()
    print(missing_values, statistics)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    df['Close'].plot(ax=ax[0], color='blue')
    ax[0].set_title('Closing Price Over Time')
    ax[0].set_ylabel('Price ($)')

    df['Volume'].plot(ax=ax[1], color='green')
    ax[1].set_title('Trading Volume Over Time')
    ax[1].set_ylabel('Volume')

    plt.tight_layout()
    plt.show()

    # Seasonality Analysis
    decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=30)
    fig = decomposition.plot()
    plt.show()

    # Volatility Analysis
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=252).std() * np.sqrt(252)
    df[['Close', 'Volatility']].plot(subplots=True, color='blue', figsize=(8, 6))
    plt.show()

    # Moving Average Analysis
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df[['Close', 'MA10', 'MA50']].plot(label='Tesla', figsize=(10,8))
    plt.show()

    # Return Analysis
    df['Return'] = df['Close'].pct_change()
    df[['Return']].plot(label='Return', figsize=(10,8))
    plt.show()

    # Prophet
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Visualize the forecast
    model.plot(forecast)
    model.plot_components(forecast)
    plt.show()

if __name__ == "__main__":
    main()