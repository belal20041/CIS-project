import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    # Clip negative predictions to avoid log errors
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def forecast_ml_model(series, horizon, max_lag, model):
    """
    Forecast future values using an iterative multi-step approach with lag features.
    """
    series = pd.Series(series).reset_index(drop=True)
    n = len(series)
    if n <= 1 or horizon >= n:
        # Not enough data to forecast; repeat last value
        return np.array([series.iloc[-1]] * horizon)
    end_train = n - horizon - 1
    lag = min(max_lag, end_train) if end_train > 0 else 1
    # Prepare lag features DataFrame
    data = {}
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = series.shift(i)
    data['y'] = series.values
    df_lag = pd.DataFrame(data).dropna().reset_index(drop=True)
    if len(df_lag) == 0:
        return np.array([series.iloc[-1]] * horizon)
    # Use all available data except last horizon rows for training
    train_df = df_lag.iloc[:-horizon] if horizon > 0 else df_lag
    X_train = train_df.drop('y', axis=1).values
    y_train = train_df['y'].values
    # Fit model
    model.fit(X_train, y_train)
    # Iterative forecasting
    last_vals = series.iloc[end_train - lag + 1 : end_train + 1].tolist()
    preds = []
    for _ in range(horizon):
        X_input = np.array(last_vals[-lag:]).reshape(1, -1)
        yhat = model.predict(X_input)[0]
        preds.append(yhat)
        last_vals.append(yhat)
    return np.array(preds)

def forecast_xgboost(series, horizon, max_lag=12):
    model = XGBRegressor(n_estimators=100, random_state=42)
    return forecast_ml_model(series, horizon, max_lag, model)

def forecast_random_forest(series, horizon, max_lag=12):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return forecast_ml_model(series, horizon, max_lag, model)

def forecast_lightgbm(series, horizon, max_lag=12):
    model = LGBMRegressor(n_estimators=100, random_state=42)
    return forecast_ml_model(series, horizon, max_lag, model)

def forecast_arima(series, horizon):
    if len(series) <= horizon + 1:
        return np.array([series.iloc[-1]] * horizon)
    try:
        model = ARIMA(series.iloc[:-horizon], order=(1,1,1))
        model_fit = model.fit()
        preds = model_fit.forecast(steps=horizon)
        return np.array(preds)
    except Exception:
        last = series.iloc[-1]
        return np.array([last] * horizon)

def forecast_sarima(series, horizon):
    if len(series) <= horizon + 1:
        return np.array([series.iloc[-1]] * horizon)
    # Determine seasonal period based on frequency
    try:
        freq = pd.infer_freq(series.index)
    except Exception:
        freq = None
    period = 1
    if isinstance(freq, str):
        if freq.startswith('W'):
            period = 52
        elif freq.startswith('M'):
            period = 12
        elif freq.startswith('D'):
            period = 7
    try:
        model = SARIMAX(series.iloc[:-horizon], order=(1,1,1), seasonal_order=(1,1,1,period))
        model_fit = model.fit(disp=False)
        preds = model_fit.forecast(steps=horizon)
        return np.array(preds)
    except Exception:
        last = series.iloc[-1]
        return np.array([last] * horizon)

def forecast_prophet(series, horizon):
    df_prophet = series.reset_index()
    df_prophet.columns = ['ds', 'y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    try:
        m.fit(df_prophet.iloc[:-horizon])
        # Determine frequency for future dataframe
        try:
            freq = pd.infer_freq(df_prophet['ds'])
            if not isinstance(freq, str):
                freq = 'D'
        except:
            freq = 'D'
        if isinstance(freq, str) and freq.startswith('W'):
            freq = 'W'
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        forecast = m.predict(future)
        preds = forecast['yhat'][-horizon:].values
        return np.array(preds)
    except Exception:
        last = series.iloc[-1]
        return np.array([last] * horizon)

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

def forecast_lstm(series, horizon):
    arr = np.array(series).reshape(-1, 1)
    look_back = min(len(arr) - horizon - 1, 10)
    if look_back < 1:
        last = series.iloc[-1]
        return np.array([last] * horizon)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(arr)
    train_scaled = data_scaled[:-horizon]
    # Prepare training sequences
    X_train, y_train = create_sequences(train_scaled, look_back)
    if len(X_train) == 0:
        last = series.iloc[-1]
        return np.array([last] * horizon)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5)
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0, callbacks=[early_stop])
    # Iterative forecast
    input_seq = train_scaled[-look_back:].reshape(1, look_back, 1)
    preds_scaled = []
    for _ in range(horizon):
        yhat = model.predict(input_seq, verbose=0)
        preds_scaled.append(yhat[0,0])
        input_seq = np.concatenate((input_seq[:,1:,:], yhat.reshape(1,1,1)), axis=1)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
    return preds

def main():
    st.title("Sales Forecasting App with Multiple Models")
    # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to proceed.")
        return
    df = pd.read_csv(uploaded_file)
    # Select columns
    cols = list(df.columns)
    date_col = st.sidebar.selectbox("Select date column", cols, index=0)
    target_col = st.sidebar.selectbox("Select target column (to forecast)", cols, index=1)
    group_col = st.sidebar.selectbox("Select group column (optional)", ["None"] + cols, index=0)
    if group_col == "None":
        group_col = None
    # Prepare data
    df = df[[date_col, target_col] + ([group_col] if group_col else [])]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df[target_col] = df[target_col].fillna(method='ffill').fillna(0)
    # Infer full date index and frequency
    dates = df[date_col].unique()
    try:
        freq = pd.infer_freq(pd.Series(dates))
    except:
        freq = None
    if freq is None:
        diffs = np.diff(pd.to_datetime(dates).astype('datetime64[D]')).astype(int)
        if len(diffs) > 0:
            median_diff = np.median(diffs)
            if 6 <= median_diff <= 8:
                freq = 'W'
            elif 27 <= median_diff <= 31:
                freq = 'M'
            else:
                freq = 'D'
        else:
            freq = 'D'
    if isinstance(freq, str) and freq.startswith('W'):
        freq = 'W'
    full_index = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)
    # Build time series
    if group_col:
        global_series = df.groupby(date_col)[target_col].sum().reindex(full_index, fill_value=0)
    else:
        global_series = df.set_index(date_col)[target_col].reindex(full_index, fill_value=0)
    if group_col:
        groups = df[group_col].unique()
        group_series_dict = {}
        for g in groups:
            grp = df[df[group_col] == g].groupby(date_col)[target_col].sum().reindex(full_index, fill_value=0)
            group_series_dict[g] = grp
    # Parameters
    horizon = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, max_value=len(full_index)-1, value=8)
    max_lag = st.sidebar.number_input("Max lag features (ML models)", min_value=1, value=12)
    model_options = ["XGBoost", "RandomForest", "LightGBM", "ARIMA", "SARIMA", "Prophet", "LSTM"]
    selected_models = st.sidebar.multiselect("Select models to include", model_options, default=model_options)
    if not selected_models:
        st.error("Please select at least one model.")
        return
    # Global forecasting
    global_metrics = []
    global_preds = pd.DataFrame({date_col: full_index[-horizon:]})
    for model_name in selected_models:
        if model_name == "XGBoost":
            preds = forecast_xgboost(global_series, horizon, max_lag)
        elif model_name == "RandomForest":
            preds = forecast_random_forest(global_series, horizon, max_lag)
        elif model_name == "LightGBM":
            preds = forecast_lightgbm(global_series, horizon, max_lag)
        elif model_name == "ARIMA":
            preds = forecast_arima(global_series, horizon)
        elif model_name == "SARIMA":
            preds = forecast_sarima(global_series, horizon)
        elif model_name == "Prophet":
            preds = forecast_prophet(global_series, horizon)
        elif model_name == "LSTM":
            preds = forecast_lstm(global_series, horizon)
        else:
            continue
        actual = global_series.values[-horizon:]
        rmse_val = rmse(actual, preds)
        mae_val = mean_absolute_error(actual, preds)
        try:
            rmsle_val = rmsle(actual, preds)
        except:
            rmsle_val = None
        global_metrics.append({"Model": model_name, "RMSE": rmse_val, "MAE": mae_val, "RMSLE": rmsle_val})
        global_preds[model_name] = preds
    # Display global metrics
    st.subheader("Global Forecast Metrics")
    df_global_metrics = pd.DataFrame(global_metrics)
    st.dataframe(df_global_metrics.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "RMSLE": "{:.4f}"}))
    # Group forecasting metrics
    if group_col:
        group_metrics = []
        for g in groups:
            series = group_series_dict[g]
            for model_name in selected_models:
                if model_name == "XGBoost":
                    preds = forecast_xgboost(series, horizon, max_lag)
                elif model_name == "RandomForest":
                    preds = forecast_random_forest(series, horizon, max_lag)
                elif model_name == "LightGBM":
                    preds = forecast_lightgbm(series, horizon, max_lag)
                elif model_name == "ARIMA":
                    preds = forecast_arima(series, horizon)
                elif model_name == "SARIMA":
                    preds = forecast_sarima(series, horizon)
                elif model_name == "Prophet":
                    preds = forecast_prophet(series, horizon)
                elif model_name == "LSTM":
                    preds = forecast_lstm(series, horizon)
                else:
                    continue
                actual = series.values[-horizon:]
                rmse_val = rmse(actual, preds)
                mae_val = mean_absolute_error(actual, preds)
                try:
                    rmsle_val = rmsle(actual, preds)
                except:
                    rmsle_val = None
                group_metrics.append({"Group": g, "Model": model_name, "RMSE": rmse_val, "MAE": mae_val, "RMSLE": rmsle_val})
        st.subheader("Per-Group Forecast Metrics")
        df_group_metrics = pd.DataFrame(group_metrics)
        st.dataframe(df_group_metrics.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "RMSLE": "{:.4f}"}))
    # Plot global forecast vs actual
    st.subheader("Global Forecast Plot")
    plot_df = pd.DataFrame({date_col: full_index[-horizon:], "Actual": global_series.values[-horizon:]})
    for model_name in selected_models:
        plot_df[model_name] = global_preds[model_name]
    plot_df_melt = plot_df.melt(id_vars=[date_col], var_name='Series', value_name='Value')
    chart = alt.Chart(plot_df_melt).mark_line(point=True).encode(
        x=f'{date_col}:T', y='Value:Q', color='Series:N'
    )
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
