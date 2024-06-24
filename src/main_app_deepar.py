import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu 
import datetime

from gluonts.dataset.common import ListDataset
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.distribution import ZeroInflatedNegativeBinomialOutput, StudentTOutput #likelihood
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.model_averaging import ModelAveraging, SelectNBestSoftmax, SelectNBestMean

from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions



# Custom CSS function
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 

# Helper function to load CSV
def load_csv(file):
    return pd.read_csv(file)

class DataCleaner:
    def __init__(self, df):
        self.df = df
    
    def clean_data(self, date_column):
        cleaned_df = self.df.copy()
        # Convert the date column to datetime format
        cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')

        # Drop rows with missing dates
        cleaned_df.dropna(subset=[date_column], inplace=True)

        # Convert the date column to Y-M-D format
        cleaned_df[date_column] = cleaned_df[date_column].dt.strftime('%Y-%m-%d')

        return cleaned_df

# Prepare data for Prophet
def prepare_data(df, date_column, target_column):
    return df[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})

# Forecasting with Prophet
def forecast_with_prophet(cleaned_df, date_column, target_column, period, seasonality):
    df = prepare_data(cleaned_df, date_column, target_column)
    model = Prophet(yearly_seasonality=seasonality['yearly'], 
                    weekly_seasonality=seasonality['weekly'], 
                    daily_seasonality=seasonality['daily'])
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    # Plot forecast and seasonality components
    fig = plot_plotly(model, forecast)
    fig_seasonality = plot_components_plotly(model, forecast)
    return forecast, fig, fig_seasonality



# Prepare data for DeepAR
def prepare_data_deepar(df, date_column, target_column, freq='D'):
    start_date = pd.Timestamp(df[date_column].min())
    target_values = df[target_column].values
    return ListDataset([{'start': start_date, 'target': target_values}], freq=freq)

def forecast_with_deepar(cleaned_df, date_column, target_column, period, freq='D', epochs=20):
    # Split the data into training and testing sets
    train_df = cleaned_df.iloc[:-period]
    test_df = cleaned_df.iloc[-period:]

    # Prepare the dataset
    training_data = prepare_data_deepar(train_df, date_column, target_column, freq)
    testing_data = prepare_data_deepar(test_df, date_column, target_column, freq)

    # Debug: Print shape and some sample data
    print("Training data head:", train_df.head())
    print("Training data tail:", train_df.tail())
    print("Training data stats:", train_df.describe())

    # Debug: Check prepared training data
    print("Prepared training data:", list(training_data)[0])

    # Define the DeepAR estimator
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=period,
        trainer=Trainer(epochs=epochs)
    )

    # Train the model
    predictor = estimator.train(training_data)
    
    # Generate forecasts
    forecast_it, ts_it = make_evaluation_predictions(testing_data, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Debug: Check the forecast samples
    print("Forecast samples shape:", forecasts[0].samples.shape)
    print("Forecast samples (first 5):", forecasts[0].samples[:5])
    print("Forecast samples mean (first 5):", forecasts[0].samples.mean(axis=0)[:5])

    # Get the forecast data
    forecast_index = pd.date_range(start=train_df[date_column].max(), periods=period + 1, freq=freq)[1:]
    forecast_df = pd.DataFrame(forecasts[0].samples.mean(axis=0), index=forecast_index, columns=['forecast'])

    # Debug: Check forecast DataFrame
    print("Forecast DataFrame:", forecast_df.head())

    # Validation
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))
    print("Aggregate metrics:", agg_metrics)

    # Display forecast table
    forecast_table = forecast_df.reset_index().rename(columns={'index': 'date'})
    print(forecast_table)

    # Plot forecast vs actual
    fig = go.Figure()

    # Plot history
    fig.add_trace(go.Scatter(x=cleaned_df[date_column], y=cleaned_df[target_column], mode='lines', name='History'))

    # Plot forecast
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast'], mode='lines', name='Forecast'))

    # Calculate and plot confidence interval
    lower_quantile = forecasts[0].quantile(0.1)
    upper_quantile = forecasts[0].quantile(0.9)

    fig.add_trace(go.Scatter(
        x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
        y=lower_quantile.tolist() + upper_quantile[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )

    st.plotly_chart(fig)

    return forecast_df, forecasts, agg_metrics, forecast_table

def recommend_actions(forecast):
    st.write("Recommended Actions:")

    # Monthly analysis
    forecast['month'] = pd.to_datetime(forecast['ds']).dt.month
    avg_monthly = forecast.groupby('month')['yhat'].mean()
    
    # Example recommendations based on average monthly values
    max_month = avg_monthly.idxmax()
    min_month = avg_monthly.idxmin()
    
    st.write(f"Highest predicted value in month: {max_month}. Consider ramping up production or stock in this period.")
    st.write(f"Lowest predicted value in month: {min_month}. Consider running promotions or discounts during this period.")
    
    # Detecting significant changes
    forecast['yhat_diff'] = forecast['yhat'].diff()
    significant_increase = forecast[forecast['yhat_diff'] > forecast['yhat_diff'].quantile(0.95)]
    significant_decrease = forecast[forecast['yhat_diff'] < forecast['yhat_diff'].quantile(0.05)]
    
    if not significant_increase.empty:
        st.write("Significant Increases Detected:")
        st.dataframe(significant_increase[['ds', 'yhat', 'yhat_diff']], width=1200)
    if not significant_decrease.empty:
        st.write("Significant Decreases Detected:")
        st.dataframe(significant_decrease[['ds', 'yhat', 'yhat_diff']],width=1200)

    # Visualizing trends
    st.write("Trend and Seasonality Analysis:")
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend Over Time", "Seasonal Components"))

    # Plotting the trend
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')), row=1, col=1)
    
    # Plotting the seasonal components
    seasonal_components = forecast[['ds', 'yearly', 'weekly']].dropna()
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['yearly'], mode='lines', name='Yearly Seasonality', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['weekly'], mode='lines', name='Weekly Seasonality', line=dict(color='red')), row=2, col=1)

    # Update layout
    fig.update_layout(height=600, width=800, title_text="Trend and Seasonality Analysis", title_x=0.5, title_y=0.9, title_font_size=24, title_font_family='Arial', title_font_color='black')

    st.plotly_chart(fig)

def main():
    st.title("Forecast Dashboard")

    # Load CSS
    local_css("style.css")

    with st.sidebar:
        st.header("Options Menu")
        selected = option_menu(
            'IntelliSeason', ["Auto Forecast", "Compare Forecast", "History"], 
            icons=['play-btn', 'search', 'info-circle'], menu_icon='intersect', default_index=0
        )

    if selected == "Auto Forecast":
        st.subheader("Auto Forecast with Prophet")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.write(df.head())
            date_column = st.selectbox("Select date column", df.columns)
            target_column = st.selectbox("Select column to forecast", df.columns)
            period = st.number_input("Forecast Period (days)", min_value=1, value=30)
            seasonality = {
                'yearly': st.checkbox("Yearly Seasonality", value=True),
                'weekly': st.checkbox("Weekly Seasonality", value=True),
                'daily': st.checkbox("Daily Seasonality", value=False)
            }
            if st.button("Run Forecast"):
                cleaner = DataCleaner(df)
                cleaned_df = cleaner.clean_data(date_column)
                forecast, fig, fig_seasonality = forecast_with_prophet(cleaned_df, date_column, target_column, period, seasonality)
                st.plotly_chart(fig)
                st.plotly_chart(fig_seasonality)
                recommend_actions(forecast)

    elif selected == "Compare Forecast":
        st.subheader("Compare Forecast")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.write(df.head())
            date_column = st.selectbox("Select date column", df.columns)
            target_column = st.selectbox("Select column to forecast", df.columns)
            period = st.number_input("Forecast Period (days)", min_value=1, value=30)
            model_choice = st.selectbox("Select Model", ["ARIMA", "CNN-QR", "DeepAR+", "ETS", "NPTS", "Prophet"])
            if st.button("Run Forecast"):
                cleaner = DataCleaner(df)
                cleaned_df = cleaner.clean_data(date_column)
                forecast_df, forecasts, agg_metrics, forecast_table = forecast_with_deepar(cleaned_df, date_column, target_column, period)
                st.write(forecasts)
                st.write(forecast_df)

                # Validation
                st.write("Aggregate metrics:", agg_metrics)

                # Display forecast table
                st.dataframe(forecast_table)

                # Plot forecast vs actual
                plt.figure(figsize=(10, 6))
                plt.plot(cleaned_df[date_column], cleaned_df[target_column], label='History')
        
                # recommend_actions(forecast)
                # TODO: Add code to compare between forecasts
                

    elif selected == "History":
        st.subheader("History (Coming Soon)")

if __name__ == '__main__':
    main()

