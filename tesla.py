import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Tesla Stock Predictor",
    layout="wide"
)

st.title("Tesla Stock Price Prediction")
st.write("LSTM-based Tesla Stock Forecasting")

# -----------------------------------
# Load Model
# -----------------------------------

model = load_model("tesla_lstm_model.keras")

# -----------------------------------
# Upload CSV
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Tesla Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Preprocessing
    # -------------------------------

    data = df[['Close']]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # -------------------------------
    # Prediction Days
    # -------------------------------

    days = st.selectbox(
        "Select Forecast Days",
        [1, 5, 10]
    )

    # -------------------------------
    # Forecast Button
    # -------------------------------

    if st.button("Predict"):

        future_input = scaled_data[-60:].flatten().tolist()

        predictions = []

        for i in range(days):

            x_input = np.array(
                future_input[-60:]
            ).reshape(1, 60, 1)

            pred = model.predict(
                x_input,
                verbose=0
            )[0][0]

            predictions.append(pred)

            future_input.append(pred)

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        st.subheader(f"Next {days} Day Prediction")

        pred_df = pd.DataFrame({
            "Day": np.arange(1, days + 1),
            "Predicted Close Price": predictions.flatten()
        })

        st.dataframe(pred_df)

        # ---------------------------
        # Plot Forecast
        # ---------------------------

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            pred_df["Day"],
            pred_df["Predicted Close Price"],
            marker='o'
        )

        ax.set_xlabel("Future Days")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Tesla Forecast")

        st.pyplot(fig)

        st.success("Prediction Completed")