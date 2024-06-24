import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_option_menu import option_menu

# Custom CSS function
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Helper function to load CSV
def load_csv(file):
    try:
        df = pd.read_csv(file)
        st.write("CSV file loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

class DataCleaner:
    def __init__(self, df):
        self.df = df
    
    def clean_data(self, date_column):
        try:
            cleaned_df = self.df.copy()
            cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
            cleaned_df.dropna(subset=[date_column], inplace=True)
            st.write("Data cleaned successfully.")
            return cleaned_df
        except Exception as e:
            st.error(f"Error cleaning data: {e}")
            return None

# Prepare data for Prophet
def prepare_data(df, date_column, target_column, regressors):
    try:
        st.write(f"Columns in DataFrame: {df.columns.tolist()}")
        st.write(f"Selected columns: {date_column}, {target_column}, {regressors}")
        cols = [date_column, target_column] + regressors
        prepared_df = df[cols].rename(columns={date_column: 'ds', target_column: 'y'})
        st.write("Data prepared for Prophet model.")
        return prepared_df
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def forecast_regressor_improved(df, period):
    try:
        # Ensure the dataframe has the correct column names for Prophet
        df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
        
        # Ensure 'ds' is a datetime column
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df.dropna(subset=['ds'], inplace=True)

        # Fit the Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        
        # Create future dataframe and make predictions
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        
        # Return only the forecasted values for the future periods
        forecasted_values = forecast['yhat'][-period:].values
        return forecasted_values
    except Exception as e:
        st.error(f"Error forecasting regressor: {e}")
        return np.array([])

def generate_future_regressors(cleaned_df, future_periods, regressors, improved=False):
    future_regressors = {}
    for regressor in regressors:
        if improved:
            future_regressor_values = forecast_regressor_improved(cleaned_df[['Date', regressor]].rename(columns={'Date': 'ds', regressor: 'y'}), future_periods)
        else:
            future_regressor_values = forecast_regressor(cleaned_df[regressor], future_periods)
        
        if future_regressor_values.size > 0:
            future_regressors[regressor] = future_regressor_values
            st.write(f"Future values for regressor {regressor}: {future_regressor_values}")
        else:
            st.error(f"Failed to generate future values for regressor {regressor}.")
    return future_regressors

def forecast_with_prophet_improved(cleaned_df, date_column, target_column, period, regressors):
    df = prepare_data(cleaned_df, date_column, target_column, regressors)
    if df is None:
        return None, None, None

    df['floor'] = 0
    
    model = Prophet(daily_seasonality=True)
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    
    future_regressors = generate_future_regressors(cleaned_df, period, regressors, improved=True)
    for regressor in regressors:
        historical_regressor_values = list(cleaned_df[regressor])
        combined_regressor_values = historical_regressor_values + list(future_regressors[regressor])
        future[regressor] = combined_regressor_values[:len(future)]  # Ensure correct length alignment

    st.write(f"Future dataframe with regressors:\n{future.tail(10)}")
    forecast = model.predict(future)
    st.write(f"Forecast:\n{forecast.tail(10)}")
    
    fig = plot_plotly(model, forecast)
    fig_seasonality = plot_components_plotly(model, forecast)
    return forecast, fig, fig_seasonality

def recommend_actions(forecast):
    try:
        st.write("Recommended Actions:")
        forecast['month'] = pd.to_datetime(forecast['ds']).dt.month
        avg_monthly = forecast.groupby('month')['yhat'].mean()
        max_month = avg_monthly.idxmax()
        min_month = avg_monthly.idxmin()
        st.write(f"Highest predicted value in month: {max_month}. Consider ramping up production or stock in this period.")
        st.write(f"Lowest predicted value in month: {min_month}. Consider running promotions or discounts during this period.")
        forecast['yhat_diff'] = forecast['yhat'].diff()
        significant_increase = forecast[forecast['yhat_diff'] > forecast['yhat_diff'].quantile(0.95)]
        significant_decrease = forecast[forecast['yhat_diff'] < forecast['yhat_diff'].quantile(0.05)]
        if not significant_increase.empty:
            st.write("Significant Increases Detected:")
            st.dataframe(significant_increase[['ds', 'yhat', 'yhat_diff']], width=1200)
        if not significant_decrease.empty:
            st.write("Significant Decreases Detected:")
            st.dataframe(significant_decrease[['ds', 'yhat', 'yhat_diff']], width=1200)
        
        st.write("Trend Analysis:")
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Trend Over Time"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')), row=1, col=1)
        fig.update_layout(height=600, width=800, title_text="Trend Analysis", title_x=0.5, title_y=0.9, title_font_size=24, title_font_family='Arial', title_font_color='black')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in recommending actions: {e}")

def calculate_alternative_metrics(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        smape = 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        return mae, rmse, smape
    except Exception as e:
        st.error(f"Error calculating alternative metrics: {e}")
        return np.inf, np.inf, np.inf

def plot_actual_vs_predicted(y_true, y_pred):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y_true))), y=y_true, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name='Predicted'))
        fig.update_layout(title='Actual vs Predicted Values', xaxis_title='Time', yaxis_title='Values')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting actual vs. predicted values: {e}")

def main():
    st.title("Forecast Dashboard")
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
            if df is not None:
                st.write(df.head())
                date_column = st.selectbox("Select date column", df.columns)
                target_column = st.selectbox("Select column to forecast", df.columns)
                filter_column = st.selectbox("Select column to filter by (optional)", [None] + list(df.columns))
                if filter_column:
                    filter_values = st.multiselect(f"Select specific values from {filter_column}", df[filter_column].unique())
                else:
                    filter_values = []

                period = st.number_input("Forecast Period (days)", min_value=1, value=30)
                regressors = st.multiselect("Select regressors to include", [col for col in df.columns if col not in [date_column, target_column]])
                
                if st.button("Run Forecast"):
                    cleaner = DataCleaner(df)
                    cleaned_df = cleaner.clean_data(date_column)
                    
                    if filter_column and filter_values:
                        cleaned_df = cleaned_df[cleaned_df[filter_column].isin(filter_values)]
                    
                    st.write("Filtered DataFrame:")
                    st.write(cleaned_df)
                    forecast, fig, fig_seasonality = forecast_with_prophet_improved(cleaned_df, date_column, target_column, period, regressors)
                    
                    if forecast is not None:
                        st.plotly_chart(fig)
                        st.plotly_chart(fig_seasonality)
                        recommend_actions(forecast)
                        
                        y_true = cleaned_df[target_column][-period:].values
                        y_pred = forecast['yhat'][-period:].values
                        mae, rmse, smape = calculate_alternative_metrics(y_true, y_pred)
                        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, sMAPE: {smape:.2f}%")
                        plot_actual_vs_predicted(y_true, y_pred)

    elif selected == "Compare Forecast":
        st.subheader("Compare Forecast")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            if df is not None:
                st.write(df.head())
                date_column = st.selectbox("Select date column", df.columns)
                target_column = st.selectbox("Select column to forecast", df.columns)
                period = st.number_input("Forecast Period (days)", min_value=1, value=30)
                model_choice = st.selectbox("Select Model", ["ARIMA", "CNN-QR", "DeepAR+", "ETS", "NPTS", "Prophet"])
                if st.button("Run Forecast"):
                    # Add code to compare forecasts
                    pass

    elif selected == "History":
        st.subheader("History (Coming Soon)")

if __name__ == '__main__':
    main()