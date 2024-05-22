import os

import streamlit as st
import pandas as pd
from prophet import Prophet

# Set the directory to save uploaded files
UPLOAD_DIR = "uploads"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

st.title("Lets make some predictions")

# File uploader widget
uploaded_file = st.file_uploader("Upload dataset CSV file", type="csv")

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

    # Ensure the DataFrame has the correct structure
    if len(df.columns) != 2:
        st.error("The CSV file must have exactly two columns: 'date' and 'value'.")
    else:
        # Prepare the data for Prophet
        # df['date'] = pd.to_datetime(df['date'])
        # df = df.rename(columns={'date': 'ds', 'value': 'y'})

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Make a future dataframe for the next year
        future = model.make_future_dataframe(periods=365)

        # Predict the future values
        forecast = model.predict(future)

        # Display the forecast
        st.write("Forecast:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot the forecast
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Plot the forecast components
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
