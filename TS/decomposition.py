import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./airline-passengers.csv')

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')

    result = seasonal_decompose(df['Passengers'], model='multiplicative')

    plt.rcParams.update({'figure.figsize': (10,10)})
    result.plot()
    plt.show()

    result2 = seasonal_decompose(df['Passengers'], model='additive')

    plt.rcParams.update({'figure.figsize': (10,10)})
    result2.plot()
    plt.show()

if __name__ == "__main__":
    main()