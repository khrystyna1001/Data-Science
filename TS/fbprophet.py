import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./airline-passengers.csv')

    df = df.rename(columns={'Passengers': 'y', 'Month': 'ds'})
    df['y_orig'] = df['y']

    # transform
    df['y'] = np.log(df['y'])
    df['y_log']=df['y']
    df['y']=df['y_orig']

    # instantiate prophet
    model = Prophet()
    model.fit(df)

    # forecast
    future_data = model.make_future_dataframe(periods=12, freq='ME')
    forecast_data = model.predict(future_data)

    # plot
    model.plot(forecast_data)
    model.plot_components(forecast_data)
    plt.show()

if __name__ == "__main__":
    main()