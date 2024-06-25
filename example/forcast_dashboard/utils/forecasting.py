
import pandas as pd
import numpy as np
from prophet import Prophet

def prepare_data(df, date_column, target_column, additional_columns):
    df = df.rename(columns={date_column: 'ds', target_column: 'y'})
    return df[['ds', 'y'] + additional_columns]

def split_data(df, date_column, train_size=0.8):
    df = df.sort_values(by=date_column)
    split_idx = int(len(df) * train_size)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    return train_df, test_df

def forecast_with_prophet(cleaned_df, date_column, target_column, period, seasonality, additional_columns):
    df = prepare_data(cleaned_df, date_column, target_column, additional_columns)
    train_df, test_df = split_data(df, 'ds')
    model = Prophet(yearly_seasonality=seasonality['yearly'], 
                    weekly_seasonality=seasonality['weekly'], 
                    daily_seasonality=seasonality['daily'])
    
    for col in additional_columns:
        model.add_regressor(col)
    
    model.fit(train_df)
    future = model.make_future_dataframe(periods=period)
    
    for col in additional_columns:
        future[col] = df[col].sum()  # assuming mean of the column for future values
    
    forecast = model.predict(future)
    return forecast, model, train_df, test_df

def validate_forecast(model, train_df, test_df):
    forecast = model.predict(test_df[['ds']])
    actual = test_df['y'].values
    predicted = forecast['yhat'].values
    percentage_errors = np.abs((actual - predicted) / actual) * 100
    min_error = np.min(percentage_errors)
    max_error = np.max(percentage_errors)
    return min_error, max_error, actual, predicted

# def validate_forecast(model, train_df, test_df):
#     forecast = model.predict(test_df[['ds']])
#     actual = test_df['y'].values
#     predicted = forecast['yhat'].values
#     error = np.mean(np.abs(predicted - actual))
#     return error, actual, predicted

