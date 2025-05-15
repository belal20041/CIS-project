import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

# Sidebar upload
st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv"], key="train")
test_file = st.sidebar.file_uploader("Upload Testing Data", type=["csv"], key="test")

# Load data
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df = df.sort_values('Date')
    return df

train_data = load_csv(train_file) if train_file else None
test_data = load_csv(test_file) if test_file else None

# Tabs
tab1, tab2, tab3 = st.tabs(["Training", "Testing", "Future Forecast"])

# --- Training Tab ---
with tab1:
    st.header("Model Training")

    if train_data is not None:
        st.subheader("Training Data")
        st.dataframe(train_data.head())

        # Convert to numeric time for regression
        train_data['Time'] = (train_data['Date'] - train_data['Date'].min()).dt.days
        X_train = train_data[['Time']]
        y_train = train_data['Sales']

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Plot
        st.subheader("Model Fit")
        plt.figure(figsize=(10, 4))
        plt.plot(train_data['Date'], y_train, label="Actual Sales")
        plt.plot(train_data['Date'], model.predict(X_train), label="Predicted Sales")
        plt.legend()
        st.pyplot(plt)

        st.success("Model trained successfully.")
    else:
        st.warning("Upload training data to train the model.")

# --- Testing Tab ---
with tab2:
    st.header("Model Testing")

    if test_data is not None and train_data is not None:
        st.subheader("Testing Data")
        st.dataframe(test_data.head())

        # Use the same time reference
        test_data['Time'] = (test_data['Date'] - train_data['Date'].min()).dt.days
        X_test = test_data[['Time']]
        y_test = test_data['Sales']

        y_pred = model.predict(X_test)

        # Plot
        st.subheader("Predictions vs Actual")
        plt.figure(figsize=(10, 4))
        plt.plot(test_data['Date'], y_test, label="Actual")
        plt.plot(test_data['Date'], y_pred, label="Predicted")
        plt.legend()
        st.pyplot(plt)

        mae = mean_absolute_error(y_test, y_pred)
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    else:
        st.warning("Upload both training and testing data.")

# --- Future Forecast Tab ---
with tab3:
    st.header("Future Forecast")

    if train_data is not None:
        forecast_days = st.number_input("Days to Forecast", min_value=1, max_value=365, value=30)

        last_date = train_data['Date'].max()
        last_time = (last_date - train_data['Date'].min()).days

        future_times = list(range(last_time + 1, last_time + 1 + forecast_days))
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_preds = model.predict(pd.DataFrame({'Time': future_times}))

        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': future_preds})
        st.subheader("Forecast Results")
        st.dataframe(forecast_df)

        # Plot
        st.subheader("Forecast Plot")
        plt.figure(figsize=(10, 4))
        plt.plot(forecast_df['Date'], forecast_df['Predicted Sales'], label="Forecast", color='green')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Train the model first.")
