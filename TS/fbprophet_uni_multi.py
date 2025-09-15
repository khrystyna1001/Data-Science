import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    pd.set_option('display.max_column', None)
    df_test = pd.read_csv('./DailyDelhiClimateTest.csv')
    df_train = pd.read_csv('./DailyDelhiClimateTrain.csv')

    df = df_train
    df_train = df_train.rename(columns={'meantemp': 'y', 'date':'ds'})
    df_train['y_orig'] = df_train['y']
    df_train['y'] = np.log(df_train['y'])

    # UNI
    model = Prophet()
    model.fit(df_train)

    future_data = model.make_future_dataframe(periods=113, freq='D')
    forecast_data = model.predict(future_data)

    model.plot(forecast_data)
    model.plot_components(forecast_data)
    plt.show()

    forecast_data_orig = forecast_data 
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])

    df_train['y_log']=df_train['y']
    df_train['y']=df_train['y_orig']

    # plot
    final_df = pd.DataFrame(forecast_data_orig)
    import plotly.graph_objs as go
    import plotly.offline as py

    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=df_train["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])

    rows = final_df["yhat"].tail(113)
    rows = rows.reset_index()
    rows.pop('index')

    plt.plot(df_test['meantemp'], color = 'blue')
    plt.plot(rows, color='red')
    plt.show()




    # MULTI
    df_test = pd.read_csv('./DailyDelhiClimateTest.csv')
    df_train = pd.read_csv('./DailyDelhiClimateTrain.csv')

    df = df_train
    df_train = df_train.rename(columns={'meantemp': 'y', 'date':'ds'})
    df_train['y_orig'] = df_train['y']
    df_train['y'] = np.log(df_train['y'])

    print(df_train.head(5))

    model_new = Prophet() #instantiate Prophet
    model_new.add_regressor('humidity')
    model_new.add_regressor('wind_speed')
    model_new.add_regressor('meanpressure')

    model_new.fit(df_train)

    df_test = df_test.rename(columns={'date': 'ds'})

    forecast_data_new = model_new.predict(df_test)

    model_new.plot(forecast_data_new)
    model_new.plot_components(forecast_data_new)
    plt.show()

    forecast_data_orig_new = forecast_data_new
    forecast_data_orig_new['yhat'] = np.exp(forecast_data_orig_new['yhat'])
    forecast_data_orig_new['yhat_lower'] = np.exp(forecast_data_orig_new['yhat_lower'])
    forecast_data_orig_new['yhat_upper'] = np.exp(forecast_data_orig_new['yhat_upper'])

    df_train['y_log']=df_train['y']
    df_train['y']=df_train['y_orig']

    # plot
    final_df_new = pd.DataFrame(forecast_data_orig_new)

    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart_new = go.Scatter(y=df_train["y_orig"], name= 'Actual')
    predict_chart_new = go.Scatter(y=final_df_new["yhat"], name= 'Predicted')
    predict_chart_upper_new = go.Scatter(y=final_df_new["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower_new = go.Scatter(y=final_df_new["yhat_lower"], name= 'Predicted Lower')
    py.plot([actual_chart_new, predict_chart_new, predict_chart_upper_new, predict_chart_lower_new])


if __name__ == "__main__":
    main()