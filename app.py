from datetime import date
import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Set up the title and stock list
st.title('Stock Trend Prediction')
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up the date range for stock data
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Select the stock
selected_stock = st.selectbox("Select dataset for prediction", tech_list)

# Set up the number of years for prediction
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load the machine learning model
def load_model_from_file():
    model = load_model('Stock_Prices.h5')  # Make sure 'Stock_Prices.h5' is in the same directory as your script
    return model

# Load stock data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    return data

ticker_data = load_data(selected_stock)

# Make predictions using the loaded model
def make_predictions(model, data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    stock_prices = ticker_data['Close'].values

# Normalize the data
    scaled_stock_prices = scaler.transform(stock_prices.reshape(-1, 1))

# Create sequences for prediction
    X_test = []

# Use the same window size as you used for training
    for i in range(60, len(scaled_stock_prices)):
        X_test.append(scaled_stock_prices[i-60:i, 0])

# Convert the X_test list to a numpy array
    X_test = np.array(X_test)

# Reshape the data to fit the model input shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
    predictions = model.predict(X_test)

# Inverse transform the predictions to get the actual stock prices
    predictions = scaler.inverse_transform(predictions)

    # Create a dataframe with the date and predicted prices
    prediction_dates = ticker_data.index[60:]
    predictions_df = pd.DataFrame(data={'Date': prediction_dates, 'Predicted Price': predictions.flatten()})
    return predictions

# Load stock data
stock_data = load_data(selected_stock)

# Load machine learning model
ml_model = load_model_from_file()

# Make predictions
predicted_prices = make_predictions(ml_model, stock_data)

# Display the results
st.subheader("Data till 2024")
st.write(stock_data)

st.subheader("Predicted Stock Prices")
predicted_dates = pd.date_range(start=end, periods=period)
predicted_data = pd.DataFrame(data={'Date': predicted_dates, 'Predicted Price': predicted_prices.flatten()})
st.write(predicted_data)

