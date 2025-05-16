import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ================== App Configuration ==================
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("Time Series Forecasting Application")

# Initialize session state for data and models if not already
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'forecasts' not in st.session_state:
    st.session_state['forecasts'] = {}

# ================== Function Definitions ==================

def train_prophet_model(df):
    """
    Train Prophet model on dataframe with columns ['ds','y'].
    """
    try:
        model = Prophet()
        model.fit(df)
        return model
    except Exception as e:
        st.error(f"Prophet training error: {e}")
        return None

def forecast_prophet_model(model, periods):
    """
    Forecast using Prophet model for given number of periods (integer).
    """
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_df = forecast_df.tail(periods).reset_index(drop=True)
    return forecast_df

def train_arima_model(series, order):
    """
    Train ARIMA model on series with specified order (p,d,q).
    """
    try:
        model = sm.tsa.ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"ARIMA training error: {e}")
        return None

def forecast_arima_model(model_fit, steps):
    """
    Forecast future values using ARIMA fitted model.
    """
    try:
        forecast = model_fit.forecast(steps=steps)
        if isinstance(forecast, (pd.Series, pd.DataFrame)):
            return forecast
        else:
            return pd.Series(forecast)
    except Exception as e:
        st.error(f"ARIMA forecasting error: {e}")
        return None

def train_sarima_model(series, order, seasonal_order):
    """
    Train SARIMA (SARIMAX) model with given orders.
    seasonal_order = (P,D,Q,s)
    """
    try:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        st.error(f"SARIMA training error: {e}")
        return None

def forecast_sarima_model(model_fit, steps):
    """
    Forecast future values using SARIMA fitted model.
    """
    try:
        forecast = model_fit.forecast(steps=steps)
        if isinstance(forecast, (pd.Series, pd.DataFrame)):
            return forecast
        else:
            return pd.Series(forecast)
    except Exception as e:
        st.error(f"SARIMA forecasting error: {e}")
        return None

def create_supervised_data(series, n_lags):
    """
    Convert a series to supervised learning format with given lags.
    Returns feature matrix X and target vector y.
    """
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_tree_model(X, y, model_type, **params):
    """
    Train a tree-based regression model (LightGBM, XGBoost, RandomForest).
    """
    try:
        if model_type == 'LightGBM':
            model = lgb.LGBMRegressor(**params)
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(**params)
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(**params)
        else:
            st.error("Invalid tree model type.")
            return None
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"{model_type} training error: {e}")
        return None

def forecast_with_tree(model, series, n_lags, steps):
    """
    Forecast future values using a trained tree-based model.
    """
    history = list(series[-n_lags:])
    preds = []
    for _ in range(steps):
        if len(history) < n_lags:
            st.error("Not enough data in history for given lags.")
            return None
        x_input = np.array(history[-n_lags:]).reshape(1, -1)
        try:
            yhat = model.predict(x_input)[0]
        except Exception as e:
            st.error(f"Forecast error: {e}")
            return None
        preds.append(yhat)
        history.append(yhat)
    return np.array(preds)

def train_lstm_model(series, n_lags, n_epochs, n_units):
    """
    Train an LSTM model for time series forecasting.
    """
    try:
        # Scale series to [0,1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = np.array(series).reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        # Create supervised dataset
        X, y = create_supervised_data(scaled.flatten(), n_lags)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        # Define LSTM network
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_lags, 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X, y, epochs=n_epochs, batch_size=1, verbose=0)
        return model, scaler
    except Exception as e:
        st.error(f"LSTM training error: {e}")
        return None, None

def forecast_lstm_model(model, scaler, series, n_lags, steps):
    """
    Forecast future values using trained LSTM model.
    """
    try:
        values = np.array(series).reshape(-1, 1)
        scaled = scaler.transform(values)
        history = list(scaled.flatten()[-n_lags:])
        preds = []
        for _ in range(steps):
            X_input = np.array(history[-n_lags:]).reshape((1, n_lags, 1))
            yhat_scaled = model.predict(X_input, verbose=0)
            preds.append(yhat_scaled[0][0])
            history.append(yhat_scaled[0][0])
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return preds
    except Exception as e:
        st.error(f"LSTM forecasting error: {e}")
        return None

# ================== Tab 1: Data Upload ==================
tab1, tab2, tab3 = st.tabs(["Data Upload", "Model Training", "Prediction"])

with tab1:
    st.header("1. Data Upload")
    st.write("Upload a CSV file containing your time series data. It should have a date/time column and a value column.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)  # Read uploaded CSV into DataFrame
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            df_raw = None
        if df_raw is not None:
            st.write("Preview of uploaded data:")
            st.dataframe(df_raw.head())
            date_col = st.selectbox("Select the date/time column", options=list(df_raw.columns))
            value_col = st.selectbox("Select the value column", options=list(df_raw.columns))
            if st.button("Load Data"):
                try:
                    df_temp = df_raw[[date_col, value_col]].dropna()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])  # Convert to datetime
                    df_temp = df_temp.sort_values(by=date_col)
                    df_temp.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
                    st.session_state['data'] = df_temp
                    # Clear any previously trained models and forecasts
                    st.session_state['models'].clear()
                    st.session_state['forecasts'].clear()
                    st.success("Data successfully loaded into the app.")
                    st.write(df_temp.describe())
                    st.line_chart(df_temp.set_index('ds')['y'])
                except Exception as e:
                    st.error(f"Error processing data: {e}")

# ================== Tab 2: Model Training ==================
with tab2:
    st.header("2. Model Training")
    if st.session_state['data'] is None:
        st.warning("Please upload and load data in the Data Upload tab first.")
    else:
        data_df = st.session_state['data']
        st.write(f"Training data loaded with {len(data_df)} records.")
        # Model selection
        model_option = st.selectbox(
            "Select Forecasting Model",
            options=["Prophet", "ARIMA", "SARIMA", "LightGBM", "XGBoost", "RandomForest", "LSTM"]
        )
        # Display hyperparameters based on model
        if model_option == "Prophet":
            st.write("Prophet has no hyperparameters to set.")
        # ARIMA parameters
        if model_option == "ARIMA":
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
            d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=0)
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=0)
        # SARIMA parameters
        if model_option == "SARIMA":
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1, key="p_sarima")
            d = st.number_input("Differencing order (d)", min_value=0, max_value=2, value=0, key="d_sarima")
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=0, key="q_sarima")
            P = st.number_input("Seasonal AR order (P)", min_value=0, max_value=2, value=1, key="P_sarima")
            D = st.number_input("Seasonal differencing (D)", min_value=0, max_value=1, value=0, key="D_sarima")
            Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=0, key="Q_sarima")
            S = st.number_input("Seasonal period (s)", min_value=1, max_value=24, value=12, key="S_sarima")
        # Tree-based model parameters
        if model_option in ["LightGBM", "XGBoost", "RandomForest"]:
            n_lags = st.number_input("Number of lag observations to use", min_value=1, max_value=len(data_df)-1, value=1)
            n_estimators = st.number_input("Number of trees/estimators", min_value=1, max_value=500, value=100)
        # LSTM parameters
        if model_option == "LSTM":
            n_lags = st.number_input("Number of lag observations to use", min_value=1, max_value=len(data_df)-1, value=1, key="lstm_lag")
            n_epochs = st.number_input("Number of training epochs", min_value=1, max_value=1000, value=50, key="lstm_epochs")
            n_units = st.number_input("Number of LSTM units", min_value=1, max_value=100, value=50, key="lstm_units")

        # Train button
        if st.button("Train Model"):
            series = data_df['y']
            # ================= Prophet Training =================
            if model_option == "Prophet":
                if len(series) < 3:
                    st.error("Not enough data for Prophet. Need at least 3 data points.")
                else:
                    model = train_prophet_model(data_df)
                    if model:
                        st.session_state['models']["Prophet"] = model
                        st.success("Prophet model trained successfully.")
            # ================= ARIMA Training =================
            if model_option == "ARIMA":
                order = (p, d, q)
                if len(series) < max(order) + 2:
                    st.warning(f"Data length is short for ARIMA order {order}. Adjusting order.")
                    order = (min(p, len(series)-2), d, min(q, len(series)-2))
                    st.write(f"Adjusted ARIMA order: {order}")
                model_fit = train_arima_model(series, order)
                if model_fit:
                    st.session_state['models']["ARIMA"] = model_fit
                    st.success(f"ARIMA model (order={order}) trained successfully.")
            # ================= SARIMA Training =================
            if model_option == "SARIMA":
                order = (p, d, q)
                seasonal_order = (P, D, Q, S)
                if len(series) < sum(order) + sum(seasonal_order[:3]) + 1:
                    st.warning("Data length is small for SARIMA parameters. Consider simpler parameters.")
                model_fit = train_sarima_model(series, order, seasonal_order)
                if model_fit:
                    st.session_state['models']["SARIMA"] = model_fit
                    st.success(f"SARIMA model (order={order}, seasonal_order={seasonal_order}) trained successfully.")
            # ================= Tree-based Model Training =================
            if model_option in ["LightGBM", "XGBoost", "RandomForest"]:
                if n_lags >= len(series):
                    st.warning(f"Number of lags too large. Adjusting to {len(series)-1}.")
                    n_lags = len(series)-1
                X, y = create_supervised_data(series.values, n_lags)
                if len(X) == 0:
                    st.error("Not enough data points for the given lags.")
                else:
                    model_fit = train_tree_model(X, y, model_option, n_estimators=n_estimators)
                    if model_fit:
                        st.session_state['models'][model_option] = (model_fit, n_lags)
                        st.success(f"{model_option} model trained successfully.")
            # ================= LSTM Training =================
            if model_option == "LSTM":
                if n_lags >= len(series):
                    st.warning(f"Number of lags too large. Adjusting to {len(series)-1}.")
                    n_lags = len(series) - 1
                if len(series) - n_lags < 1:
                    st.error("Not enough data points to train LSTM with these lags.")
                else:
                    model_fit, scaler = train_lstm_model(series.values, n_lags, n_epochs, n_units)
                    if model_fit:
                        st.session_state['models']["LSTM"] = (model_fit, scaler, n_lags)
                        st.success("LSTM model trained successfully.")

# ================== Tab 3: Prediction ==================
with tab3:
    st.header("3. Prediction")
    if st.session_state['data'] is None:
        st.warning("Please upload data and train a model before forecasting.")
    else:
        if not st.session_state['models']:
            st.warning("No trained model available. Please train a model in the Model Training tab.")
        else:
            model_option = st.selectbox("Select model for prediction", options=list(st.session_state['models'].keys()))
            horizon = st.number_input("Forecast horizon (number of periods)", min_value=1, value=10)
            if st.button("Generate Forecast"):
                series = st.session_state['data']['y']
                # ================= Prophet Forecast =================
                if model_option == "Prophet":
                    model = st.session_state['models']['Prophet']
                    future_df = forecast_prophet_model(model, horizon)
                    st.write("Forecast results (Prophet):")
                    st.dataframe(future_df)
                    # Download CSV
                    csv = future_df.to_csv(index=False).encode()
                    st.download_button("Download Forecast as CSV", csv, "forecast_prophet.csv", "text/csv")
                # ================= ARIMA Forecast =================
                if model_option == "ARIMA":
                    model_fit = st.session_state['models']['ARIMA']
                    forecast_vals = forecast_arima_model(model_fit, horizon)
                    if forecast_vals is not None:
                        df_forecast = pd.DataFrame({
                            'step': np.arange(1, horizon+1),
                            'forecast': forecast_vals.flatten()
                        })
                        st.write("Forecast results (ARIMA):")
                        st.dataframe(df_forecast)
                        csv = df_forecast.to_csv(index=False).encode()
                        st.download_button("Download Forecast as CSV", csv, "forecast_arima.csv", "text/csv")
                # ================= SARIMA Forecast =================
                if model_option == "SARIMA":
                    model_fit = st.session_state['models']['SARIMA']
                    forecast_vals = forecast_sarima_model(model_fit, horizon)
                    if forecast_vals is not None:
                        df_forecast = pd.DataFrame({
                            'step': np.arange(1, horizon+1),
                            'forecast': forecast_vals.flatten()
                        })
                        st.write("Forecast results (SARIMA):")
                        st.dataframe(df_forecast)
                        csv = df_forecast.to_csv(index=False).encode()
                        st.download_button("Download Forecast as CSV", csv, "forecast_sarima.csv", "text/csv")
                # ================= Tree-based Forecast =================
                if model_option in ["LightGBM", "XGBoost", "RandomForest"]:
                    model_fit, n_lags = st.session_state['models'][model_option]
                    preds = forecast_with_tree(model_fit, series.values, n_lags, horizon)
                    if preds is not None:
                        df_forecast = pd.DataFrame({
                            'step': np.arange(1, horizon+1),
                            'forecast': preds.flatten()
                        })
                        st.write(f"Forecast results ({model_option}):")
                        st.dataframe(df_forecast)
                        csv = df_forecast.to_csv(index=False).encode()
                        st.download_button("Download Forecast as CSV", csv, f"forecast_{model_option.lower()}.csv", "text/csv")
                # ================= LSTM Forecast =================
                if model_option == "LSTM":
                    model_fit, scaler, n_lags = st.session_state['models']['LSTM']
                    preds = forecast_lstm_model(model_fit, scaler, series.values, n_lags, horizon)
                    if preds is not None:
                        df_forecast = pd.DataFrame({
                            'step': np.arange(1, horizon+1),
                            'forecast': preds.flatten()
                        })
                        st.write("Forecast results (LSTM):")
                        st.dataframe(df_forecast)
                        csv = df_forecast.to_csv(index=False).encode()
                        st.download_button("Download Forecast as CSV", csv, "forecast_lstm.csv", "text/csv")
