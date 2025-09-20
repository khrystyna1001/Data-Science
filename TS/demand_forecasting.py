import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def main():
    pd.set_option('display.max_column', None)
    df = pd.read_csv('./demand.csv')

    # EDA
    missing_values = df.isnull().sum()

    median_price_per_sku = df.groupby('sku_id')['total_price'].median()
    df['total_price'].fillna(df['sku_id'].map(median_price_per_sku), inplace=True)

    # print(df.isnull().sum())

    # make TS
    df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')
    weekly_data = df.groupby('week')['units_sold'].sum().reset_index()
    print(weekly_data.head())

    # plot
    # Plot histograms for categorical columns and the target variable
    categorical_cols = ['store_id', 'sku_id', 'is_featured_sku', 'is_display_sku', 'units_sold']

    for col in categorical_cols:
        plt.figure(figsize=(10,4))
        plt.hist(df[col], bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Boxplot for numerical columns
    numerical_cols = ['total_price', 'base_price']

    for col in numerical_cols:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    # Check the correlation between features
    correlation = df.corr()

    # Plot a heatmap of the correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    # Plot the time series of 'units_sold'
    plt.figure(figsize=(15, 6))
    plt.plot(weekly_data['week'], weekly_data['units_sold'])
    plt.title('Time Series of Units Sold')
    plt.xlabel('Week')
    plt.ylabel('Units Sold')
    plt.show()

    # decomposition
    weekly_data.set_index('week', inplace=True)
    decomposition = seasonal_decompose(weekly_data['units_sold'], period=52)
    plt.figure(figsize=(12,8))

    # Trend
    plt.subplot(411)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')

    # Seasonality
    plt.subplot(412)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')

    # Residuals
    plt.subplot(413)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')

    # Original
    plt.subplot(414)
    plt.plot(weekly_data['units_sold'], label='Original')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    weekly_data['units_sold_pct_change'] = weekly_data['units_sold'].pct_change()

    # Calculate the standard deviation of these percentage changes
    volatility = weekly_data['units_sold_pct_change'].std()
    print(volatility)

    # Calculate the 4-week and 52-week moving averages
    weekly_data['4_week_MA'] = weekly_data['units_sold'].rolling(window=4).mean()
    weekly_data['52_week_MA'] = weekly_data['units_sold'].rolling(window=52).mean()

    # Plot the original time series and the moving averages
    plt.figure(figsize=(15, 6))

    plt.plot(weekly_data['units_sold'], label='Original')
    plt.plot(weekly_data['4_week_MA'], label='4-Week Moving Average')
    plt.plot(weekly_data['52_week_MA'], label='52-Week Moving Average')

    plt.title('Time Series of Units Sold with Moving Averages')
    plt.xlabel('Week')
    plt.ylabel('Units Sold')
    plt.legend()

    plt.show()

    from pandas.plotting import autocorrelation_plot

    # Plot the autocorrelation
    plt.figure(figsize=(12, 5))
    autocorrelation_plot(weekly_data['units_sold'])
    plt.show()

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # Split the data into train and test sets
    train_data = weekly_data['units_sold'][:int(0.8*(len(weekly_data)))]
    test_data = weekly_data['units_sold'][int(0.8*(len(weekly_data))):]

    # Fit the model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()

    # Resample the data to a weekly frequency
    weekly_data_resampled = weekly_data['units_sold'].resample('W').sum()

    # Split the resampled data into train and test sets
    train_data_resampled = weekly_data_resampled[:int(0.8*(len(weekly_data_resampled)))]
    test_data_resampled = weekly_data_resampled[int(0.8*(len(weekly_data_resampled))):]

    # Fit the model to the resampled data
    model_resampled = ExponentialSmoothing(train_data_resampled, seasonal='add', seasonal_periods=52).fit()

    # Generate predictions on the resampled data
    predictions_resampled = model_resampled.predict(start=test_data_resampled.index[0], end=test_data_resampled.index[-1])

    # Calculate the root mean squared error (RMSE) on the resampled data
    rmse_resampled = sqrt(mean_squared_error(test_data_resampled, predictions_resampled))
    print("RMSE resampled", rmse_resampled)

    # Plot the original time series and the forecasted values
    plt.figure(figsize=(15, 6))

    plt.plot(train_data_resampled, label='Training')
    plt.plot(test_data_resampled, label='Test')
    plt.plot(predictions_resampled, label='Forecast')

    plt.title('Time Series of Units Sold with Forecast')
    plt.xlabel('Week')
    plt.ylabel('Units Sold')
    plt.legend()

    plt.show()

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # Load the data
    data = pd.read_csv('demand.csv')

    # Convert the 'week' column to datetime format and set it as the index
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    # Aggregate the data by week
    weekly_data = data['units_sold'].resample('W').sum()

    # Split the data into train and test sets
    train_data = weekly_data[:int(0.8*(len(weekly_data)))]
    test_data = weekly_data[int(0.8*(len(weekly_data))):]

    # Fit a Holt-Winters model
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()

    # Generate predictions from the Holt-Winters model
    hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])

    # Fit an ARIMA model
    arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()

    # Generate predictions from the ARIMA model
    arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])

    # Prepare the data for Prophet
    prophet_data = weekly_data.reset_index()
    prophet_data.columns = ['ds', 'y']

    # Split the data into train and test sets
    train_data_prophet = prophet_data[:int(0.8*(len(prophet_data)))]
    test_data_prophet = prophet_data[int(0.8*(len(prophet_data))):]

    # Fit a Prophet model
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_data_prophet)

    # Generate predictions from the Prophet model
    prophet_future = prophet_model.make_future_dataframe(periods=len(test_data_prophet))
    prophet_predictions = prophet_model.predict(prophet_future)

    # Calculate the RMSE for each model
    hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))
    arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))
    prophet_rmse = sqrt(mean_squared_error(test_data_prophet['y'], prophet_predictions['yhat'][-len(test_data_prophet):]))

    # Print the RMSE for each model
    print(f'Holt-Winters RMSE: {hw_rmse}')
    print(f'ARIMA RMSE: {arima_rmse}')
    print(f'Prophet RMSE: {prophet_rmse}')

    # Plot the original data and the predictions from each model
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data, label='Original')
    plt.plot(hw_predictions, label='Holt-Winters')
    plt.plot(arima_predictions, label='ARIMA')
    plt.plot(prophet_predictions['ds'], prophet_predictions['yhat'], label='Prophet')
    plt.legend()
    plt.show()

    # Compare the models and print the name of the best model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters', 'ARIMA', 'Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]

    print(f'The best model is: {best_model}')

    # Plot the original data and the predictions from each model
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data, label='Original')
    plt.plot(hw_predictions, label='Holt-Winters')
    #plt.plot(arima_predictions, label='ARIMA')
    plt.plot(prophet_predictions['ds'], prophet_predictions['yhat'], label='Prophet')
    plt.legend()
    plt.show()

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # Load the data
    data = pd.read_csv('demand.csv')

    # Convert the 'week' column to datetime format and set it as the index
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    # Aggregate the data by week
    weekly_data = data['units_sold'].resample('W').sum()

    # Split the data into train and test sets
    train_data = weekly_data[:int(0.8*(len(weekly_data)))]
    test_data = weekly_data[int(0.8*(len(weekly_data))):]

    # Fit a Holt-Winters model
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()

    # Generate predictions from the Holt-Winters model
    hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])

    # Fit an ARIMA model
    arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()

    # Generate predictions from the ARIMA model
    arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])

    # Prepare the data for Prophet
    prophet_data = weekly_data.reset_index()
    prophet_data.columns = ['ds', 'y']

    # Split the data into train and test sets
    train_data_prophet = prophet_data[:int(0.8*(len(prophet_data)))]
    test_data_prophet = prophet_data[int(0.8*(len(prophet_data))):]

    # Fit a Prophet model
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_data_prophet)

    # Generate predictions from the Prophet model for the test period
    prophet_future = prophet_model.make_future_dataframe(periods=len(test_data))
    prophet_predictions = prophet_model.predict(prophet_future)

    # Calculate the RMSE for each model
    hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))
    arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))
    prophet_rmse = sqrt(mean_squared_error(test_data, prophet_predictions['yhat'][-len(test_data):]))

    # Print the RMSE for each model
    print(f'Holt-Winters RMSE: {hw_rmse}')
    print(f'ARIMA RMSE: {arima_rmse}')
    print(f'Prophet RMSE: {prophet_rmse}')

    # Plot the original data and the predictions from each model
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data, label='Original')
    plt.plot(hw_predictions, label='Holt-Winters')
    plt.plot(arima_predictions, label='ARIMA')
    plt.plot(test_data.index, prophet_predictions['yhat'][-len(test_data):].values, label='Prophet')
    plt.legend()
    plt.show()

    # Compare the models and print the name of the best model
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters', 'ARIMA', 'Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]

    print(f'The best model is: {best_model}')

    df = data

    # Aggregate the data by week
    weekly_data_fb = df['units_sold'].resample('W').sum()

    # Prepare the data for Prophet
    fbprophet_data = weekly_data_fb.reset_index()
    fbprophet_data.columns = ['ds', 'y']

    train_data_prophet = fbprophet_data[:int(0.8*(len(fbprophet_data)))]
    test_data_prophet = fbprophet_data[int(0.8*(len(fbprophet_data))):]

    # Fit a Prophet model
    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_data_prophet)

    # Generate predictions from the Prophet model for the test period
    prophet_future = prophet_model.make_future_dataframe(periods=len(test_data_prophet))
    prophet_predictions = prophet_model.predict(prophet_future)

    # Extract the predicted and actual values
    prophet_predicted = prophet_predictions[-len(test_data_prophet):]['yhat']
    prophet_actual = test_data_prophet['y']

    # Calculate the RMSE
    prophet_rmse = np.sqrt(mean_squared_error(prophet_actual, prophet_predicted))
    print('Prophet RMSE:', prophet_rmse)

    final_df = pd.DataFrame(prophet_predictions)
    import plotly.graph_objs as go
    import plotly.offline as py
    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=train_data_prophet["y"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])


                

if __name__ == "__main__":
    main()