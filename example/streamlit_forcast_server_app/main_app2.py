import os
import streamlit as st
import pandas as pd
from prophet import Prophet

# Set the directory to save uploaded files
UPLOAD_DIR = "uploads"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Custom CSS for modern, colorful styling and sidebar
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin: auto;
        /* max-width: 1000px; */
    }
    .title {
        font-size: 2.5em;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .stNumberInput input, .sidebar .stTextInput input, .sidebar .stRadio label, .sidebar .stFileUploader label {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #4A90E2;
        color: white;
        border-radius: 5px;
        padding: 10px;
        border: none;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    .stAlert {
        margin-top: 20px;
        color: black !important;  /* Ensures alert text is black */
    }
    .stDataFrame {
        margin-top: 20px;
        max-width: 100%; 
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Let\'s Make Some Predictions</div>', unsafe_allow_html=True)

# Sidebar with input parameters
with st.sidebar:
    st.header("Input Parameters")

    # Ask the user if they want a general forecast or a forecast based on a specific column
    forecast_type = st.radio("Choose forecast type:", ("General Forecast", "Specific Column Forecast"))

    # Ask the user for the number of columns in the dataset
    num_columns = st.number_input("Number of columns:", min_value=1, step=1)

    # Ask for the column name for forecasting
    forecast_col = st.text_input("Column for forecasting:", help="Enter the column name")

    specific_col = None
    if forecast_type == "Specific Column Forecast":
        # Ask for the specific column name (e.g., region)
        specific_col = st.text_input("Specific column for filtering:", help="Enter the specific column name (e.g., region)")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload dataset CSV file", type="csv")

# Main area for displaying results
if uploaded_file is not None:
    # Save uploaded file to the specified directory
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File '{uploaded_file.name}' saved to '{UPLOAD_DIR}' successfully!")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df.head())

    if forecast_col:
        # Ensure the specified column exists in the DataFrame
        if forecast_col in df.columns:
            st.success(f"Data has the specified column '{forecast_col}' for forecasting.")

            if forecast_type == "Specific Column Forecast" and specific_col:
                if specific_col in df.columns:
                    # Filter data based on the specific column
                    unique_values = df[specific_col].unique()
                    selected_value = st.selectbox(f"Select {specific_col}:", unique_values)
                    df = df[df[specific_col] == selected_value]
                else:
                    st.error(f"The specified column '{specific_col}' is not found in the dataset.")
                    st.stop()

            # Clean the data for Prophet
            df = df[['Date', forecast_col]].rename(columns={'Date': 'ds', forecast_col: 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            st.success("Data cleaned for forecasting.")

            # Initialize and fit the Prophet model
            model = Prophet()
            model.fit(df)

            # Make a future dataframe for the next year
            future = model.make_future_dataframe(periods=365)

            # Predict the future values
            forecast = model.predict(future)

            # Display the forecast
            st.write("Forecast:")
            st.write(forecast[['ds', 'yhat']], width=800)

            # Plot the forecast
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Plot the forecast components
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
        else:
            st.error("The specified column name is not found in the dataset.")
    else:
        st.error("Please specify a valid column name for forecasting.")






