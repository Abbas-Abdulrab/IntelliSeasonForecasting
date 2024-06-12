import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
import streamlit as st

def plot_forecast(model, forecast):
    fig = plot_plotly(model, forecast)
    return fig

def plot_seasonality(model, forecast):
    fig_seasonality = plot_components_plotly(model, forecast)
    return fig_seasonality

def plot_trend_and_seasonality(forecast):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend Over Time", "Seasonal Components"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')), row=1, col=1)
    seasonal_components = forecast[['ds', 'yearly', 'weekly']].dropna()
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['yearly'], mode='lines', name='Yearly Seasonality', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['weekly'], mode='lines', name='Weekly Seasonality', line=dict(color='red')), row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="Trend and Seasonality Analysis", title_x=0.5, title_y=0.9, title_font_size=24, title_font_family='Arial', title_font_color='black')
    st.plotly_chart(fig)

def plot_validation(dates, actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title="Validation Results", xaxis_title="Date", yaxis_title="Values")
    # st.plotly_chart(fig)
    return fig

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
    plot_trend_and_seasonality(forecast)



# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from prophet.plot import plot_plotly, plot_components_plotly
# import streamlit as st

# def plot_forecast(model, forecast):
#     fig = plot_plotly(model, forecast)
#     return fig

# def plot_seasonality(model, forecast):
#     fig_seasonality = plot_components_plotly(model, forecast)
#     return fig_seasonality

# def plot_trend_and_seasonality(forecast):
#     fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend Over Time", "Seasonal Components"))
#     fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='blue')), row=1, col=1)
#     seasonal_components = forecast[['ds', 'yearly', 'weekly']].dropna()
#     fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['yearly'], mode='lines', name='Yearly Seasonality', line=dict(color='green')), row=2, col=1)
#     fig.add_trace(go.Scatter(x=seasonal_components['ds'], y=seasonal_components['weekly'], mode='lines', name='Weekly Seasonality', line=dict(color='red')), row=2, col=1)
#     fig.update_layout(height=600, width=800, title_text="Trend and Seasonality Analysis", title_x=0.5, title_y=0.9, title_font_size=24, title_font_family='Arial', title_font_color='black')
#     st.plotly_chart(fig)

# def recommend_actions(forecast):
#     st.write("Recommended Actions:")
#     forecast['month'] = pd.to_datetime(forecast['ds']).dt.month
#     avg_monthly = forecast.groupby('month')['yhat'].mean()
#     max_month = avg_monthly.idxmax()
#     min_month = avg_monthly.idxmin()
#     st.write(f"Highest predicted value in month: {max_month}. Consider ramping up production or stock in this period.")
#     st.write(f"Lowest predicted value in month: {min_month}. Consider running promotions or discounts during this period.")
#     forecast['yhat_diff'] = forecast['yhat'].diff()
#     significant_increase = forecast[forecast['yhat_diff'] > forecast['yhat_diff'].quantile(0.95)]
#     significant_decrease = forecast[forecast['yhat_diff'] < forecast['yhat_diff'].quantile(0.05)]
#     if not significant_increase.empty:
#         st.write("Significant Increases Detected:")
#         st.dataframe(significant_increase[['ds', 'yhat', 'yhat_diff']], width=1200)
#     if not significant_decrease.empty:
#         st.write("Significant Decreases Detected:")
#         st.dataframe(significant_decrease[['ds', 'yhat', 'yhat_diff']], width=1200)
#     plot_trend_and_seasonality(forecast)
