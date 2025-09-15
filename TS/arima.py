import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./airline-passengers.csv')

    dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
    df = pd.read_csv('./airline-passengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)

    # Convert to time series
    ts = df['Passengers']

    # Stationarity test
    def test_stationary(timeseries):
        rolmean = pd.Series(timeseries).rolling(window=12).mean()
        rolstd = pd.Series(timeseries).rolling(window=12).std()

        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

        #Advanced Dickey-Fuller Test
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    
    print("INITIAL")
    test_stationary(ts)
    plt.show()

    # Transformations
    print("LOG TEST")
    ts_log = np.log(ts)
    test_stationary(ts_log)
    plt.show()

    print("DLOG TEST")
    dts_log = np.log(ts_log)
    test_stationary(dts_log)
    plt.show()

    print("MOVING AVERAGE")
    ma = pd.Series(ts_log).rolling(window=12).mean()
    plt.plot(ts_log)
    plt.plot(ma, color="red")
    plt.show()

    print("LOG MA DIFF")
    ts_log_ma_diff = ts_log - ma
    ts_log_ma_diff.dropna(inplace=True)
    print(ts_log_ma_diff.head(5))

    print("EW AVERAGE")
    expwa = ts_log.ewm(span=12).mean()
    plt.plot(ts_log)
    plt.plot(expwa, color="red")

    ts_log_ewma_diff = ts_log - expwa
    test_stationary(ts_log_ewma_diff)
    plt.show()

    print("LOG DIFF")
    ts_log_diff = ts_log - ts_log.shift()
    plt.plot(ts_log_diff)
    ts_log_diff.dropna(inplace=True)
    test_stationary(ts_log_diff)
    plt.show()

    # Decomposition
    decomposition = seasonal_decompose(ts_log)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    test_stationary(ts_log_decompose)
    plt.show()

    # ACF / PACF
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    plt.subplot(121)    
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    
    ts_values=ts_log.values

    X = ts_values
    size = int(len(X) * 0.667)
    train, test = X[0:size], X[size:len(X)]

    # Grid Search
    # Define ranges for p, q, and d
    p_values = [1,2]
    d_values = [0,1]
    q_values = [1,2]

    best_rmse, best_p, best_d, best_q = np.inf, None, None, None
    history = [x for x in train]
    # make predictions
    predictions = list()
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    for t in range(len(test)):
                        # Fit the model
                        model = ARIMA(history, order=order)
                        model_fit = model.fit()
                        yhat = model_fit.forecast()[0]
                        predictions.append(yhat)
                        history.append(test[t])

                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(test, predictions))

                    # Update best RMSE and parameter values
                    if rmse < best_rmse:
                            best_rmse, best_p, best_d, best_q = rmse, p, d, q

                except:
                    continue

    print(f"Best RMSE: {best_rmse}")
    print(f"Best p: {best_p}")
    print(f"Best d: {best_d}")
    print(f"Best q: {best_q}")

    # Final Forecast
    history = [x for x in train]
    predictions = list()
    #test.reset_index()
    for t in range(len(test)):
        try:
            model = ARIMA(history, order=(4,1,2))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        except (ValueError, LinAlgError):
            pass
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    rmse = mean_squared_error(test, predictions)**0.5
    print('Test MSE: %.3f' % rmse)


    from math import sqrt
    rms = sqrt(mean_squared_error(test, predictions))

    pyplot.plot(test, color = 'blue', label='test')   
    pyplot.plot(predictions, color='red', label='pred')
    pyplot.show()

if __name__ == "__main__":
    main()