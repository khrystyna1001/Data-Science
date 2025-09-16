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
            

if __name__ == "__main__":
    main()