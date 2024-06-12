import streamlit as st
from streamlit_option_menu import option_menu
from utils.data_loading import load_csv
from utils.data_cleaning import DataCleaner
from utils.forecasting import forecast_with_prophet, validate_forecast
from utils.visualization import plot_forecast, plot_seasonality, plot_validation

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def toggle_theme():
    
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

def main():
    st.title("Forecast Dashboard")

    # Check if theme is already set, if not, set it to light mode
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    # Load CSS based on the selected theme
    if st.session_state.theme == "light":
        st.markdown('<link rel="stylesheet" type="text/css" href="style.css">', unsafe_allow_html=True)
    else:
        st.markdown('<link rel="stylesheet" type="text/css" href="style.css">', unsafe_allow_html=True)

    # Toggle theme button
    toggle_theme()    
    # st.title("Forecast Dashboard")
    # local_css("style.css")

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
            additional_columns = st.multiselect("Select additional columns for forecasting", df.columns.difference([date_column, target_column]))
            period = st.number_input("Forecast Period (days)", min_value=1, value=30)
            seasonality = {
                'yearly': st.checkbox("Yearly Seasonality", value=True),
                'weekly': st.checkbox("Weekly Seasonality", value=True),
                'daily': st.checkbox("Daily Seasonality", value=False)
            }
            
            # Add filter options
            filter_column = st.selectbox("Select column to filter by", df.columns)
            filter_values = df[filter_column].unique().tolist()
            filter_value = st.selectbox("Select value to filter by", filter_values)
            
            if st.button("Run Forecast"):
                cleaner = DataCleaner(df)
                filtered_df = cleaner.filter_data(filter_column, filter_value)
                cleaned_df = cleaner.clean_data(date_column)
                aggregated_df = cleaner.aggregate_data(date_column, target_column, additional_columns)
                forecast, model, train_df, test_df = forecast_with_prophet(aggregated_df, date_column, 'y', period, seasonality, additional_columns)
                st.plotly_chart(plot_forecast(model, forecast))
                st.plotly_chart(plot_seasonality(model, forecast))
                
                error, actual, predicted = validate_forecast(model, train_df, test_df)
                st.write(f"Validation MAE: {error}")
                st.plotly_chart(plot_validation(test_df['ds'], actual, predicted))

    elif selected == "Compare Forecast":
        st.subheader("Compare Forecast")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.write(df.head())
            date_column = st.selectbox("Select date column", df.columns)
            target_column = st.selectbox("Select column to forecast", df.columns)
            additional_columns = st.multiselect("Select additional columns for forecasting", df.columns.difference([date_column, target_column]))
            period = st.number_input("Forecast Period (days)", min_value=1, value=30)
            models_to_compare = st.multiselect("Select Models to Compare", ["ARIMA", "Prophet"], default=["ARIMA", "Prophet"], key="models")
            
            # Add filter options
            filter_column = st.selectbox("Select column to filter by", df.columns)
            filter_values = df[filter_column].unique().tolist()
            filter_value = st.selectbox("Select value to filter by", filter_values)
            
            if len(models_to_compare) == 2 and st.button("Run Forecast"):
                cleaner = DataCleaner(df)
                filtered_df = cleaner.filter_data(filter_column, filter_value)
                cleaned_df = cleaner.clean_data(date_column)
                aggregated_df = cleaner.aggregate_data(date_column, target_column, additional_columns)
                
                for model_choice in models_to_compare:
                    if model_choice == "ARIMA":
                        arima_model, arima_forecast = forecast_with_arima(aggregated_df, date_column, 'y', period)
                    elif model_choice == "Prophet":
                        seasonality = {
                            'yearly': st.checkbox("Yearly Seasonality", value=True),
                            'weekly': st.checkbox("Weekly Seasonality", value=True),
                            'daily': st.checkbox("Daily Seasonality", value=False)
                        }
                        prophet_model, prophet_forecast = forecast_with_prophet(aggregated_df, date_column, 'y', period, seasonality, additional_columns)
                
                if "ARIMA" in models_to_compare:
                    st.write("ARIMA Forecast:")
                    st.write(arima_forecast)
                if "Prophet" in models_to_compare:
                    st.write("Prophet Forecast:")
                    st.write(prophet_forecast)
                
    elif selected == "History":
        st.subheader("History (Coming Soon)")

if __name__ == '__main__':
    main()
