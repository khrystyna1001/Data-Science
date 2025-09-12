import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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

if __name__ == "__main__":
    main()