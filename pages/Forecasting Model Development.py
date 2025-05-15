import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("Sales Forecasting Dashboard")

# Initialize session state for data and predictions
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

# Sidebar: Data upload
st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Load data into session state if uploaded
if train_file is not None:
    try:
        st.session_state['train_df'] = pd.read_csv(train_file)
    except Exception as e:
        st.sidebar.error(f"Error reading training file: {e}")
if test_file is not None:
    try:
        st.session_state['test_df'] = pd.read_csv(test_file)
    except Exception as e:
        st.sidebar.error(f"Error reading test file: {e}")

# If both datasets are uploaded, validate and display previews
if st.session_state['train_df'] is not None and st.session_state['test_df'] is not None:
    train_df = st.session_state['train_df']
    test_df = st.session_state['test_df']
    
    # Validate required columns
    required_cols = {'id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion'}
    missing_train = required_cols - set(train_df.columns)
    missing_test = required_cols - set(test_df.columns)
    if missing_train:
        st.sidebar.error(f"Training data missing columns: {missing_train}")
    if missing_test:
        st.sidebar.error(f"Test data missing columns: {missing_test}")
    
    # Only proceed if no missing columns
    if not missing_train and not missing_test:
        # Data previews
        st.subheader("Training Data Preview")
        st.dataframe(train_df.head())
        st.subheader("Test Data Preview")
        st.dataframe(test_df.head())
        
        # Sidebar: Model selection
        st.sidebar.header("Model Selection")
        model_option = st.sidebar.selectbox(
            "Choose Forecasting Model",
            ("XGBoost", "ARIMA", "SARIMA", "Prophet", "LSTM")
        )
        
        # Data preprocessing function
        def preprocess_data(train, test):
            # Combine train and test for consistent encoding/lag features
            data = pd.concat([train, test], sort=False).reset_index(drop=True)
            # Convert date column to datetime and extract features
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['day'] = data['date'].dt.day
            data['dayofweek'] = data['date'].dt.dayofweek
            # Label encode categorical variables
            le_store = LabelEncoder()
            le_family = LabelEncoder()
            data['store_nbr_enc'] = le_store.fit_transform(data['store_nbr'])
            data['family_enc'] = le_family.fit_transform(data['family'])
            # Create lag features grouped by store and family
            data.sort_values(['store_nbr', 'family', 'date'], inplace=True)
            data['sales_lag1'] = data.groupby(['store_nbr','family'])['sales'].shift(1)
            data['sales_lag7'] = data.groupby(['store_nbr','family'])['sales'].shift(7)
            data.fillna(0, inplace=True)
            # Scale sales (useful for LSTM)
            scaler = MinMaxScaler()
            data['sales_scaled'] = scaler.fit_transform(data[['sales']])
            return data, le_store, le_family, scaler

        # Forecasting functions for each model
        def run_xgboost(train, test):
            features = ['store_nbr_enc', 'family_enc', 'month', 'day', 'dayofweek', 'sales_lag1', 'sales_lag7']
            X_train = train[features]
            y_train = train['sales']
            X_test = test[features]
            model = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return preds
        
        def run_arima(train, test):
            train_ts = train[['date','sales']].set_index('date')
            try:
                model = ARIMA(train_ts['sales'], order=(1,1,1))
                model_fit = model.fit()
                # Forecast for length of test set
                forecast = model_fit.forecast(len(test))
                return forecast.values
            except Exception as e:
                st.error(f"ARIMA error: {e}")
                return []
        
        def run_sarima(train, test):
            train_ts = train[['date','sales']].set_index('date')
            try:
                model = sm.tsa.statespace.SARIMAX(
                    train_ts['sales'], 
                    order=(1,1,1), 
                    seasonal_order=(1,1,1,7)
                )
                model_fit = model.fit(disp=False)
                forecast = model_fit.forecast(len(test))
                return forecast.values
            except Exception as e:
                st.error(f"SARIMA error: {e}")
                return []
        
        def run_prophet(train, test):
            df_prophet = train.rename(columns={'date': 'ds', 'sales': 'y'})[['ds','y']]
            try:
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=len(test))
                forecast = model.predict(future)
                # Extract the forecast for the test period
                return forecast[['yhat']].tail(len(test)).values.flatten()
            except Exception as e:
                st.error(f"Prophet error: {e}")
                return []
        
        def run_lstm(train, test):
            features = ['sales_lag1', 'sales_lag7']
            X_train = train[features].values.reshape((train.shape[0], len(features), 1))
            y_train = train['sales_scaled']
            X_test = test[features].values.reshape((test.shape[0], len(features), 1))
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            predictions = model.predict(X_test)
            # Inverse transform the scaled predictions
            predictions = scaler.inverse_transform(predictions)
            return predictions.flatten()
        
        # Preprocess data
        processed_data, le_store, le_family, scaler = preprocess_data(train_df, test_df)
        train_processed = processed_data[processed_data['id'].isin(train_df['id'])]
        test_processed = processed_data[processed_data['id'].isin(test_df['id'])]
        
        # Placeholder for predictions
        preds = None
        
        # Button to trigger training and prediction
        if st.button("Train and Predict"):
            with st.spinner(f"Running {model_option} model..."):
                if model_option == "XGBoost":
                    preds = run_xgboost(train_processed, test_processed)
                elif model_option == "ARIMA":
                    preds = run_arima(train_processed, test_processed)
                elif model_option == "SARIMA":
                    preds = run_sarima(train_processed, test_processed)
                elif model_option == "Prophet":
                    preds = run_prophet(train_processed, test_processed)
                elif model_option == "LSTM":
                    preds = run_lstm(train_processed, test_processed)
                else:
                    st.error("Invalid model selected.")
                
                # If predictions were made, display results
                if preds is not None:
                    results_df = test_df.copy()
                    results_df['forecast'] = preds
                    st.subheader("Forecast Results")
                    st.dataframe(results_df.head())
                    st.session_state['predictions'] = results_df
        else:
            st.info("Click 'Train and Predict' to run the forecasting model.")
        
        # Download button for predictions
        if st.session_state['predictions'] is not None:
            csv_data = st.session_state['predictions'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Forecast CSV", 
                data=csv_data, 
                file_name="forecast.csv", 
                mime="text/csv"
            )
else:
    st.info("Please upload both training and test data to proceed.")
