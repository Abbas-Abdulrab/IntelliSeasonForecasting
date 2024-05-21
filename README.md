# Time Series Forecasting API

This project is a Python 3-based API for time series forecasting using the [Darts](https://github.com/unit8co/darts) library. It supports multiple forecasting models, including Prophet. Users can upload a `dataset.csv` file, specify the column to predict, and select the forecasting model to use. The API will process the data and return the predictions.

## Features

- **Upload Dataset**: Users can upload a CSV file containing the time series data.
- **Model Selection**: Users can choose from various forecasting models, including Prophet.
- **Column Specification**: Users specify which column in the dataset they want to predict.
- **Predictions**: The API processes the data and returns the forecasted values.

## Requirements

- Python 3.8+
- [Darts](https://github.com/unit8co/darts)
- [Prophet](https://facebook.github.io/prophet/)
- Flask or FastAPI for creating the API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forecasting-api.git
   cd forecasting-api

## Install the required packages:
`pip install -r requirements.txt`

## Start the API:
`python main_app.py`

## Upload a Dataset:
Use an API client like Postman or curl to upload a dataset.csv file, specify the column to predict, and select the model.

### Example curl command:

`curl -X POST "http://localhost:5000/predict" \
-F "file=@path/to/your/dataset.csv" \
-F "column=your_column_name" \
-F "model=prophet"`

## Get Predictions:
The API will return the forecasted values based on the specified model.

## API Endpoints
TBD

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or suggestions, please contact GenZ.