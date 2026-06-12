Tesla Stock Price Prediction using SimpleRNN and LSTM
Project Overview

This project focuses on predicting Tesla (TSLA) stock closing prices using Deep Learning techniques. Since stock market data is sequential in nature, Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks are used to learn historical patterns and forecast future stock prices.

The project compares the performance of SimpleRNN and LSTM models and generates predictions for 1-day, 5-day, and 10-day future stock prices.

Problem Statement

The objective of this project is to:

Predict Tesla stock closing prices using historical stock data.
Implement and compare SimpleRNN and LSTM models.
Forecast future stock prices for:
Next 1 Day
Next 5 Days
Next 10 Days
Evaluate model performance using standard regression metrics.
Deploy the final model using Streamlit.
Dataset

The dataset contains Tesla stock market data with the following attributes:

Feature	Description
Date	Trading Date
Open	Opening Price
High	Highest Price
Low	Lowest Price
Close	Closing Price
Adj Close	Adjusted Closing Price
Volume	Number of Shares Traded
Target Variable
Close Price
Technologies Used
Programming Language
Python 3.x
Libraries
NumPy
Pandas
Matplotlib
Scikit-learn
TensorFlow / Keras
Streamlit
Project Workflow
1. Data Collection
Load Tesla stock dataset.
Explore dataset structure and statistics.
2. Data Preprocessing
Handle missing values.
Convert Date column to datetime format.
Set Date as index.
Select Close Price for prediction.
Normalize data using MinMaxScaler.
3. Exploratory Data Analysis (EDA)

Visualizations performed:

Closing Price Trend
Trading Volume Trend
Moving Average Analysis
4. Feature Engineering
Create time-series sequences using a sliding window approach.
Use previous 60 days' stock prices to predict the next day.
5. Model Development
SimpleRNN

Architecture:

SimpleRNN Layer
Dropout Layer
Dense Output Layer
LSTM

Architecture:

LSTM Layer
Dropout Layer
Dense Output Layer
6. Model Training
Adam Optimizer
Mean Squared Error (MSE) Loss Function
Early Stopping
7. Model Evaluation

Evaluation Metrics:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)

Visualization:

Actual vs Predicted Stock Prices
8. Future Forecasting

Generate forecasts for:

1-Day Prediction
5-Day Prediction
10-Day Prediction
9. Deployment

Deploy the trained LSTM model using Streamlit.
