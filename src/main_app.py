import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import datetime

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
        cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
        cleaned_df.dropna(subset=[date_column], inplace=True)
        cleaned_df[date_column] = cleaned_df[date_column].dt.strftime('%Y-%m-%d')
        return cleaned_df

# Prepare data for Prophet
def prepare_data(df, date_column, target_column):
    return df[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})

def forecast_with_prophet(cleaned_df, date_column, target_column, period, seasonality):
    df = prepare_data(cleaned_df, date_column, target_column)
    
    model = Prophet(yearly_seasonality=seasonality['yearly'], 
                    weekly_seasonality=seasonality['weekly'], 
                    daily_seasonality=seasonality['daily'])
    
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    fig = plot_plotly(model, forecast)
    fig_seasonality = plot_components_plotly(model, forecast)
    return forecast, fig, fig_seasonality

def recommend_actions(forecast):
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
    st.write("Trend and Seasonality Analysis:")
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend Over Time", "Seasonal Components"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')), row=1, col=1)
    seasonal_components = forecast[['ds', 'yearly', 'weekly']].dropna()
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['yearly'], mode='lines', name='Yearly Seasonality', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['weekly'], mode='lines', name='Weekly Seasonality', line=dict(color='red')), row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="Trend and Seasonality Analysis", title_x=0.5, title_y=0.9, title_font_size=24, title_font_family='Arial', title_font_color='black')
    st.plotly_chart(fig)

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
            st.write(df.head())
            date_column = st.selectbox("Select date column", df.columns)
            target_column = st.selectbox("Select column to forecast", df.columns)
            filter_column = st.selectbox("Select column to filter by (optional)", [None] + list(df.columns))
            if filter_column:
                filter_values = st.multiselect(f"Select specific values from {filter_column}", df[filter_column].unique())
            else:
                filter_values = []

            period = st.number_input("Forecast Period (days)", min_value=1, value=30)
            seasonality = {
                'yearly': st.checkbox("Yearly Seasonality", value=True),
                'weekly': st.checkbox("Weekly Seasonality", value=True),
                'daily': st.checkbox("Daily Seasonality", value=False)
            }
            if st.button("Run Forecast"):
                cleaner = DataCleaner(df)
                cleaned_df = cleaner.clean_data(date_column)
                
                # Apply filtering if specific values were selected
                if filter_column and filter_values:
                    cleaned_df = cleaned_df[cleaned_df[filter_column].isin(filter_values)]
                
                # Display the filtered DataFrame
                st.write("Filtered DataFrame:")
                st.write(cleaned_df)
                
                forecast, fig, fig_seasonality = forecast_with_prophet(cleaned_df, date_column, target_column, period, seasonality)
                
                if forecast is not None:
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
                # Add code to compare forecasts
                pass

    elif selected == "History":
        st.subheader("History (Coming Soon)")

if __name__ == '__main__':
    main()
