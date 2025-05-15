import streamlit as st
import pandas as pd
import numpy as np

# Modeling libraries
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Plotly for interactive charts
import plotly.graph_objects as go

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    # ensure non-negative for RMSLE calculation
    try:
        rmsle = np.sqrt(mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0)))
    except:
        rmsle = None
    return rmse, mae, rmsle

# Function for iterative forecasting using tree-based models (LightGBM, XGBoost, RandomForest)
def iterative_forecast(train_series, horizon, model, n_lags=7):
    """
    Train a model on train_series using previous n_lags as features, then forecast horizon steps iteratively.
    """
    if len(train_series) <= n_lags:
        raise ValueError("Not enough data for forecasting with given n_lags.")
    X, y = [], []
    series = np.array(train_series).astype(float)
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    X_train = np.array(X)
    y_train = np.array(y)
    model.fit(X_train, y_train)
    preds = []
    last_window = list(series[-n_lags:])
    for _ in range(horizon):
        X_input = np.array(last_window).reshape(1, -1)
        pred = model.predict(X_input)[0]
        preds.append(pred)
        last_window.pop(0)
        last_window.append(pred)
    return preds

# ARIMA forecasting
def forecast_arima(train_series, horizon, order=(1,1,0)):
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return np.array(forecast)

# SARIMA forecasting
def forecast_sarima(train_series, horizon, order=(1,1,0), seasonal_order=(1,1,0,7)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return np.array(forecast)

# Prophet forecasting
def forecast_prophet(train_df, horizon):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)
    return forecast['yhat'].values[-horizon:]

# LSTM forecasting
def forecast_lstm(train_series, horizon, n_lags=7, n_units=50, epochs=5):
    series = np.array(train_series).astype(float)
    generator = TimeseriesGenerator(series, series, length=n_lags, batch_size=1)
    if len(generator) == 0:
        raise ValueError("Not enough data for LSTM with given n_lags.")
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_lags, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=epochs, verbose=0)
    preds = []
    last_seq = series[-n_lags:]
    for _ in range(horizon):
        X_input = last_seq.reshape((1, n_lags, 1))
        pred = model.predict(X_input, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    return preds

def main():
    st.title("Sales Forecasting App with Multiple Models")

    st.sidebar.header("Configuration")

    # File upload
    data_file = st.sidebar.file_uploader("Upload sales data (CSV)", type=['csv'])
    if data_file is None:
        st.info("Please upload a CSV file to proceed.")
        return

    # Read data
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # Select columns
    cols = list(df.columns)
    date_col = st.sidebar.selectbox("Select date column", cols)
    target_col = st.sidebar.selectbox("Select sales (target) column", cols)
    group_col = st.sidebar.selectbox("Select group column (optional)", [None] + cols)

    # Forecast horizon
    forecast_horizon = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, value=7, step=1)
    # Seasonal period for SARIMA
    seasonal_period = st.sidebar.number_input("Seasonal period for SARIMA (e.g., 7 for weekly)", min_value=0, value=7, step=1)

    # Model selection
    model_options = ['LightGBM', 'XGBoost', 'LSTM', 'ARIMA', 'SARIMA', 'RandomForest', 'Prophet']
    selected_models = st.sidebar.multiselect("Select models to train", model_options)

    train_button = st.sidebar.button("Train and Forecast")

    if train_button:
        # Preprocess data
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            st.error(f"Error parsing dates: {e}")
            return

        # Filter necessary columns
        if group_col:
            df = df[[date_col, target_col, group_col]].dropna()
            df = df.sort_values([group_col, date_col])
        else:
            df = df[[date_col, target_col]].dropna()
            df = df.sort_values(date_col)

        # Prepare series
        if group_col:
            global_df = df.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col: 'sales'})
            groups = df[group_col].unique()
            group_dfs = {g: df[df[group_col] == g][[date_col, target_col]].rename(columns={target_col: 'sales'}).reset_index(drop=True) for g in groups}
        else:
            df = df.rename(columns={target_col: 'sales'})
            global_df = df.copy()
            groups = ['All']
            group_dfs = {'All': df.copy()}

        # Check enough data
        if len(global_df) <= forecast_horizon:
            st.error("Not enough data points for forecasting horizon.")
            return

        # Split train and test
        global_train = global_df.iloc[:-forecast_horizon].copy()
        global_test = global_df.iloc[-forecast_horizon:].copy()

        # Initialize results containers
        global_metrics = []
        group_metrics = []
        global_preds_df = global_test[[date_col]].copy()
        global_preds_df['Actual'] = global_test['sales'].values
        group_preds_list = []

        # Train and forecast for each selected model
        for model_name in selected_models:
            # Global forecast
            try:
                if model_name == 'LightGBM':
                    model = LGBMRegressor(n_estimators=20, n_jobs=1)
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, n_lags=min(7, len(global_train)-1))
                elif model_name == 'XGBoost':
                    model = XGBRegressor(n_estimators=20, n_jobs=1, verbosity=0)
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, n_lags=min(7, len(global_train)-1))
                elif model_name == 'RandomForest':
                    model = RandomForestRegressor(n_estimators=20, n_jobs=1)
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, n_lags=min(7, len(global_train)-1))
                elif model_name == 'ARIMA':
                    preds = forecast_arima(global_train['sales'], forecast_horizon, order=(1,1,0))
                    model = None
                elif model_name == 'SARIMA':
                    if seasonal_period > 0:
                        seasonal_order = (1, 1, 0, seasonal_period)
                    else:
                        seasonal_order = (0, 0, 0, 0)
                    preds = forecast_sarima(global_train['sales'], forecast_horizon, order=(1,1,0), seasonal_order=seasonal_order)
                    model = None
                elif model_name == 'Prophet':
                    prophet_df = global_train.rename(columns={date_col: 'ds', 'sales': 'y'})
                    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                    prophet_model.fit(prophet_df)
                    future = prophet_model.make_future_dataframe(periods=forecast_horizon, freq='D')
                    forecast = prophet_model.predict(future)
                    preds = forecast['yhat'].values[-forecast_horizon:]
                    model = prophet_model
                elif model_name == 'LSTM':
                    n_lags = min(7, len(global_train)-1)
                    preds = forecast_lstm(global_train['sales'], forecast_horizon, n_lags=n_lags, n_units=50, epochs=5)
                    model = None
                else:
                    st.warning(f"Unknown model: {model_name}")
                    continue

                # Store global predictions
                global_preds_df[model_name] = np.array(preds)
                # Compute and store metrics
                rmse, mae, rmsle = compute_metrics(global_test['sales'].values, np.array(preds))
                global_metrics.append({'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'RMSLE': rmsle})
                # Save model in session state if available
                if model is not None:
                    st.session_state.setdefault('models', {})[model_name] = model
            except Exception as e:
                st.error(f"{model_name} (global) failed: {e}")
                continue

            # Per-group forecasts
            if group_col:
                for g in groups:
                    try:
                        g_df = group_dfs[g]
                        if len(g_df) <= forecast_horizon:
                            continue
                        g_train = g_df.iloc[:-forecast_horizon].copy()
                        g_test = g_df.iloc[-forecast_horizon:].copy()
                        if model_name == 'LightGBM':
                            model_g = LGBMRegressor(n_estimators=20, n_jobs=1)
                            preds_g = iterative_forecast(g_train['sales'], forecast_horizon, model_g, n_lags=min(7, len(g_train)-1))
                        elif model_name == 'XGBoost':
                            model_g = XGBRegressor(n_estimators=20, n_jobs=1, verbosity=0)
                            preds_g = iterative_forecast(g_train['sales'], forecast_horizon, model_g, n_lags=min(7, len(g_train)-1))
                        elif model_name == 'RandomForest':
                            model_g = RandomForestRegressor(n_estimators=20, n_jobs=1)
                            preds_g = iterative_forecast(g_train['sales'], forecast_horizon, model_g, n_lags=min(7, len(g_train)-1))
                        elif model_name == 'ARIMA':
                            preds_g = forecast_arima(g_train['sales'], forecast_horizon, order=(1,1,0))
                        elif model_name == 'SARIMA':
                            if seasonal_period > 0:
                                seasonal_order = (1, 1, 0, seasonal_period)
                            else:
                                seasonal_order = (0, 0, 0, 0)
                            preds_g = forecast_sarima(g_train['sales'], forecast_horizon, order=(1,1,0), seasonal_order=seasonal_order)
                        elif model_name == 'Prophet':
                            prophet_df_g = g_train.rename(columns={date_col: 'ds', 'sales': 'y'})
                            prophet_model_g = Prophet(daily_seasonality=True, weekly_seasonality=True)
                            prophet_model_g.fit(prophet_df_g)
                            future_g = prophet_model_g.make_future_dataframe(periods=forecast_horizon, freq='D')
                            forecast_g = prophet_model_g.predict(future_g)
                            preds_g = forecast_g['yhat'].values[-forecast_horizon:]
                        elif model_name == 'LSTM':
                            n_lags = min(7, len(g_train)-1)
                            preds_g = forecast_lstm(g_train['sales'], forecast_horizon, n_lags=n_lags, n_units=50, epochs=5)
                        else:
                            continue
                        # Store group predictions
                        for i, date in enumerate(g_test[date_col].values):
                            group_preds_list.append({
                                'Group': g,
                                'Date': date,
                                'Actual': g_test['sales'].values[i],
                                model_name: preds_g[i]
                            })
                        # Store group metrics
                        rmse_g, mae_g, rmsle_g = compute_metrics(g_test['sales'].values, np.array(preds_g))
                        group_metrics.append({'Group': g, 'Model': model_name, 'RMSE': rmse_g, 'MAE': mae_g, 'RMSLE': rmsle_g})
                    except Exception as e:
                        st.warning(f"{model_name} failed for group {g}: {e}")
                        continue

        # Create DataFrames for metrics
        if global_metrics:
            global_metrics_df = pd.DataFrame(global_metrics)
            st.subheader("Global Forecast Metrics")
            st.dataframe(global_metrics_df)
        else:
            global_metrics_df = pd.DataFrame(columns=['Model','RMSE','MAE','RMSLE'])

        if group_col and group_metrics:
            group_metrics_df = pd.DataFrame(group_metrics)
            st.subheader("Per-Group Forecast Metrics")
            st.dataframe(group_metrics_df)
        else:
            group_metrics_df = pd.DataFrame(columns=['Group','Model','RMSE','MAE','RMSLE'])

        # Global forecast plot
        st.subheader("Global Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=global_test[date_col], y=global_test['sales'], mode='lines+markers', name='Actual'))
        for model_name in selected_models:
            if model_name in global_preds_df.columns:
                fig.add_trace(go.Scatter(x=global_test[date_col], y=global_preds_df[model_name], mode='lines+markers', name=model_name))
        st.plotly_chart(fig, use_container_width=True)

        # Per-group forecast plot
        if group_col:
            st.subheader("Per-Group Forecast Plot")
            df_gp = pd.DataFrame(group_preds_list)
            if not df_gp.empty:
                # Transform to wide format for plotting
                df_melt = df_gp.melt(id_vars=['Group','Date','Actual'], value_vars=[m for m in selected_models], 
                                     var_name='Model', value_name='Prediction')
                df_melt = df_melt.dropna(subset=['Prediction'])
                df_gp_wide = df_melt.pivot_table(index=['Group','Date','Actual'], columns='Model', values='Prediction').reset_index()
                group_list = list(df_gp_wide['Group'].unique())
                selected_group = st.selectbox("Select group", group_list)
                df_gp_sel = df_gp_wide[df_gp_wide['Group'] == selected_group]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_gp_sel['Date'], y=df_gp_sel['Actual'], mode='lines+markers', name='Actual'))
                for model_name in selected_models:
                    if model_name in df_gp_sel.columns:
                        fig2.add_trace(go.Scatter(x=df_gp_sel['Date'], y=df_gp_sel[model_name], mode='lines+markers', name=model_name))
                st.plotly_chart(fig2, use_container_width=True)

        # Store results in session_state
        st.session_state['global_predictions'] = global_preds_df
        st.session_state['global_metrics'] = global_metrics_df
        if group_col:
            st.session_state['group_predictions'] = df_gp_wide if not df_gp.empty else None
            st.session_state['group_metrics'] = group_metrics_df

        # Download buttons
        st.subheader("Download Results")
        if not global_preds_df.empty:
            global_preds_csv = global_preds_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Global Predictions CSV", data=global_preds_csv, file_name='global_predictions.csv', mime='text/csv')
        if group_col and not df_gp.empty:
            group_preds_csv = df_gp_wide.to_csv(index=False).encode('utf-8')
            st.download_button("Download Group Predictions CSV", data=group_preds_csv, file_name='group_predictions.csv', mime='text/csv')
        if not global_metrics_df.empty:
            metrics_global_csv = global_metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Global Metrics CSV", data=metrics_global_csv, file_name='global_metrics.csv', mime='text/csv')
        if group_col and not group_metrics_df.empty:
            metrics_group_csv = group_metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Group Metrics CSV", data=metrics_group_csv, file_name='group_metrics.csv', mime='text/csv')

if __name__ == '__main__':
    main()
