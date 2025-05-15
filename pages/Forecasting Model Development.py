import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to preprocess data
def preprocess_data(df, date_col, target_col, freq='D'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col).asfreq(freq, method='ffill').reset_index()
    
    # Handle missing values
    df[target_col] = df[target_col].interpolate(method='linear')
    
    # Handle outliers (using IQR method)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    df[target_col] = df[target_col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    
    return df

# Function to create features
def create_features(df, date_col, target_col, n_lags=7):
    df = df.copy()
    # Date-based features
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lagged features
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_7'] = df[target_col].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df[target_col].shift(1).rolling(window=7).std()
    
    return df.dropna()

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmsle = np.sqrt(mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0)))
    except:
        rmsle = None
    return rmse, mae, rmsle

# Cross-validation function
def cross_validate_model(train_series, model_func, n_splits=3, horizon=7, **kwargs):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(train_series):
        train = train_series.iloc[train_idx]
        test = train_series.iloc[test_idx][:horizon]
        try:
            preds = model_func(train, horizon, **kwargs)
            rmse, _, _ = compute_metrics(test.values, preds)
            scores.append(rmse)
        except Exception as e:
            logger.warning(f"Cross-validation fold failed: {e}")
    return np.mean(scores) if scores else np.inf

# Iterative forecasting for tree-based models
def iterative_forecast(train_series, horizon, model, feature_df, n_lags=7):
    if len(train_series) <= n_lags:
        raise ValueError("Not enough data for forecasting.")
    X = feature_df.drop(columns=['sales']).values
    y = feature_df['sales'].values
    model.fit(X, y)
    preds = []
    last_features = feature_df.iloc[-1:].copy()
    for _ in range(horizon):
        X_input = last_features.drop(columns=['sales']).values
        pred = model.predict(X_input)[0]
        preds.append(pred)
        # Update features for next step
        last_features['sales'] = pred
        for lag in range(n_lags, 0, -1):
            if lag > 1:
                last_features[f'lag_{lag-1}'] = last_features[f'lag_{lag}']
            else:
                last_features[f'lag_{lag}'] = pred
        last_features['rolling_mean_7'] = last_features[[f'lag_{i}' for i in range(1, min(8, n_lags+1))]].mean(axis=1)
        last_features['rolling_std_7'] = last_features[[f'lag_{i}' for i in range(1, min(8, n_lags+1))]].std(axis=1)
    return preds

# ARIMA forecasting with auto-order selection
def forecast_arima(train_series, horizon, order=None):
    if order is None:
        best_aic = np.inf
        best_order = (1, 1, 0)
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(train_series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        order = best_order
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    return fitted.forecast(steps=horizon)

# SARIMA forecasting
def forecast_sarima(train_series, horizon, order=(1,1,0), seasonal_order=(1,1,0,7)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=horizon)

# Prophet forecasting with additional regressors
def forecast_prophet(train_df, horizon, extra_features=None):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    if extra_features:
        for feature in extra_features:
            model.add_regressor(feature)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=horizon, freq='D')
    if extra_features:
        future = future.merge(train_df[['ds'] + extra_features], on='ds', how='left')
        future[extra_features] = future[extra_features].fillna(method='ffill')
    forecast = model.predict(future)
    return forecast['yhat'].values[-horizon:]

# LSTM forecasting with scaling and early stopping
def forecast_lstm(train_series, horizon, n_lags=7, n_units=100, epochs=20):
    scaler = StandardScaler()
    series = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    generator = TimeseriesGenerator(series, series, length=n_lags, batch_size=1)
    if len(generator) == 0:
        raise ValueError("Not enough data for LSTM.")
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_lags, 1), return_sequences=True))
    model.add(LSTM(n_units // 2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(generator, epochs=epochs, callbacks=[early_stop], verbose=0)
    preds = []
    last_seq = series[-n_lags:]
    for _ in range(horizon):
        X_input = last_seq.reshape((1, n_lags, 1))
        pred = model.predict(X_input, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def main():
    st.title("Sales Forecasting App with Enhanced Models")

    st.sidebar.header("Configuration")
    data_file = st.sidebar.file_uploader("Upload sales data (CSV)", type=['csv'])
    if data_file is None:
        st.info("Please upload a CSV file to proceed.")
        return

    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    cols = list(df.columns)
    date_col = st.sidebar.selectbox("Select date column", cols)
    target_col = st.sidebar.selectbox("Select sales (target) column", cols)
    group_col = st.sidebar.selectbox("Select group column (optional)", [None] + cols)
    forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, value=7, step=1)
    seasonal_period = st.sidebar.number_input("Seasonal period for SARIMA", min_value=0, value=7, step=1)
    n_lags = st.sidebar.number_input("Number of lags", min_value=1, value=7, step=1)
    model_options = ['LightGBM', 'XGBoost', 'LSTM', 'ARIMA', 'SARIMA', 'RandomForest', 'Prophet']
    selected_models = st.sidebar.multiselect("Select models to train", model_options)
    train_button = st.sidebar.button("Train and Forecast")

    if train_button:
        try:
            df = preprocess_data(df, date_col, target_col)
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return

        if group_col:
            df = df[[date_col, target_col, group_col]].dropna()
            df = df.sort_values([group_col, date_col])
            global_df = df.groupby(date_col)[target_col].sum().reset_index().rename(columns={target_col: 'sales'})
            groups = df[group_col].unique()
            group_dfs = {g: df[df[group_col] == g][[date_col, target_col]].rename(columns={target_col: 'sales'}) for g in groups}
        else:
            df = df.rename(columns={target_col: 'sales'})
            global_df = df[[date_col, 'sales']].copy()
            groups = ['All']
            group_dfs = {'All': global_df.copy()}

        if len(global_df) <= forecast_horizon + n_lags:
            st.error("Not enough data points for forecasting.")
            return

        global_train = global_df.iloc[:-forecast_horizon].copy()
        global_test = global_df.iloc[-forecast_horizon:].copy()
        global_train_features = create_features(global_train, date_col, 'sales', n_lags=n_lags)

        global_metrics = []
        group_metrics = []
        global_preds_df = global_test[[date_col]].copy()
        global_preds_df['Actual'] = global_test['sales'].values
        group_preds_list = []

        for model_name in selected_models:
            try:
                if model_name == 'LightGBM':
                    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1)
                    cv_score = cross_validate_model(global_train['sales'], 
                                                    lambda s, h: iterative_forecast(s, h, model, global_train_features, n_lags=n_lags), 
                                                    horizon=forecast_horizon)
                    logger.info(f"LightGBM CV RMSE: {cv_score}")
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, global_train_features, n_lags=n_lags)
                elif model_name == 'XGBoost':
                    model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1, verbosity=0)
                    cv_score = cross_validate_model(global_train['sales'], 
                                                    lambda s, h: iterative_forecast(s, h, model, global_train_features, n_lags=n_lags), 
                                                    horizon=forecast_horizon)
                    logger.info(f"XGBoost CV RMSE: {cv_score}")
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, global_train_features, n_lags=n_lags)
                elif model_name == 'RandomForest':
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1)
                    cv_score = cross_validate_model(global_train['sales'], 
                                                    lambda s, h: iterative_forecast(s, h, model, global_train_features, n_lags=n_lags), 
                                                    horizon=forecast_horizon)
                    logger.info(f"RandomForest CV RMSE: {cv_score}")
                    preds = iterative_forecast(global_train['sales'], forecast_horizon, model, global_train_features, n_lags=n_lags)
                elif model_name == 'ARIMA':
                    preds = forecast_arima(global_train['sales'], forecast_horizon)
                    cv_score = cross_validate_model(global_train['sales'], forecast_arima, horizon=forecast_horizon)
                    logger.info(f"ARIMA CV RMSE: {cv_score}")
                elif model_name == 'SARIMA':
                    seasonal_order = (1, 1, 0, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0)
                    preds = forecast_sarima(global_train['sales'], forecast_horizon, seasonal_order=seasonal_order)
                    cv_score = cross_validate_model(global_train['sales'], forecast_sarima, horizon=forecast_horizon, seasonal_order=seasonal_order)
                    logger.info(f"SARIMA CV RMSE: {cv_score}")
                elif model_name == 'Prophet':
                    prophet_df = global_train_features.rename(columns={date_col: 'ds', 'sales': 'y'})
                    extra_features = ['day_of_week', 'month', 'is_weekend', 'rolling_mean_7']
                    preds = forecast_prophet(prophet_df, forecast_horizon, extra_features=extra_features)
                    cv_score = cross_validate_model(global_train['sales'], 
                                                    lambda s, h: forecast_prophet(global_train_features.rename(columns={date_col: 'ds', 'sales': 'y'}), h, extra_features), 
                                                    horizon=forecast_horizon)
                    logger.info(f"Prophet CV RMSE: {cv_score}")
                elif model_name == 'LSTM':
                    n_lags = min(n_lags, len(global_train)-1)
                    preds = forecast_lstm(global_train['sales'], forecast_horizon, n_lags=n_lags, n_units=100, epochs=20)
                    cv_score = cross_validate_model(global_train['sales'], forecast_lstm, horizon=forecast_horizon, n_lags=n_lags)
                    logger.info(f"LSTM CV RMSE: {cv_score}")
                else:
                    continue

                global_preds_df[model_name] = np.array(preds)
                rmse, mae, rmsle = compute_metrics(global_test['sales'].values, np.array(preds))
                global_metrics.append({'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'RMSLE': rmsle, 'CV_RMSE': cv_score})

            except Exception as e:
                st.error(f"{model_name} (global) failed: {e}")
                logger.error(f"{model_name} error: {e}")
                continue

            if group_col:
                for g in groups:
                    try:
                        g_df = preprocess_data(group_dfs[g], date_col, 'sales')
                        if len(g_df) <= forecast_horizon + n_lags:
                            continue
                        g_train = g_df.iloc[:-forecast_horizon].copy()
                        g_test = g_df.iloc[-forecast_horizon:].copy()
                        g_train_features = create_features(g_train, date_col, 'sales', n_lags=n_lags)
                        if model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
                            model_g = {
                                'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1),
                                'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1, verbosity=0),
                                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1)
                            }[model_name]
                            preds_g = iterative_forecast(g_train['sales'], forecast_horizon, model_g, g_train_features, n_lags=n_lags)
                        elif model_name == 'ARIMA':
                            preds_g = forecast_arima(g_train['sales'], forecast_horizon)
                        elif model_name == 'SARIMA':
                            seasonal_order = (1, 1, 0, seasonal_period) if seasonal_period > 0 else (0, 0, 0, 0)
                            preds_g = forecast_sarima(g_train['sales'], forecast_horizon, seasonal_order=seasonal_order)
                        elif model_name == 'Prophet':
                            prophet_df_g = g_train_features.rename(columns={date_col: 'ds', 'sales': 'y'})
                            extra_features = ['day_of_week', 'month', 'is_weekend', 'rolling_mean_7']
                            preds_g = forecast_prophet(prophet_df_g, forecast_horizon, extra_features=extra_features)
                        elif model_name == 'LSTM':
                            preds_g = forecast_lstm(g_train['sales'], forecast_horizon, n_lags=n_lags, n_units=100, epochs=20)
                        else:
                            continue
                        for i, date in enumerate(g_test[date_col].values):
                            group_preds_list.append({
                                'Group': g,
                                'Date': date,
                                'Actual': g_test['sales'].values[i],
                                model_name: preds_g[i]
                            })
                        rmse_g, mae_g, rmsle_g = compute_metrics(g_test['sales'].values, np.array(preds_g))
                        group_metrics.append({'Group': g, 'Model': model_name, 'RMSE': rmse_g, 'MAE': mae_g, 'RMSLE': rmsle_g})
                    except Exception as e:
                        st.warning(f"{model_name} failed for group {g}: {e}")
                        logger.warning(f"{model_name} group {g} error: {e}")
                        continue

        # Display metrics
        if global_metrics:
            global_metrics_df = pd.DataFrame(global_metrics)
            st.subheader("Global Forecast Metrics")
            st.dataframe(global_metrics_df)
        else:
            global_metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'RMSLE', 'CV_RMSE'])

        if group_col and group_metrics:
            group_metrics_df = pd.DataFrame(group_metrics)
            st.subheader("Per-Group Forecast Metrics")
            st.dataframe(group_metrics_df)
        else:
            group_metrics_df = pd.DataFrame(columns=['Group', 'Model', 'RMSE', 'MAE', 'RMSLE'])

        # Global forecast plot
        st.subheader("Global Forecast Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=global_test[date_col], y=global_test['sales'], mode='lines+markers', name='Actual'))
        for model_name in selected_models:
            if model_name in global_preds_df.columns:
                fig.add_trace(go.Scatter(x=global_test[date_col], y=global_preds_df[model_name], mode='lines+markers', name=model_name))
        st.plotly_chart(fig, use_container_width=True)

        # Residual plot for global forecasts
        st.subheader("Global Residual Plot")
        fig_res = go.Figure()
        for model_name in selected_models:
            if model_name in global_preds_df.columns:
                residuals = global_test['sales'].values - global_preds_df[model_name].values
                fig_res.add_trace(go.Scatter(x=global_test[date_col], y=residuals, mode='lines+markers', name=f"{model_name} Residuals"))
        st.plotly_chart(fig_res, use_container_width=True)

        # Per-group forecast plot
        if group_col:
            st.subheader("Per-Group Forecast Plot")
            df_gp = pd.DataFrame(group_preds_list)
            if not df_gp.empty:
                df_melt = df_gp.melt(id_vars=['Group', 'Date', 'Actual'], value_vars=[m for m in selected_models], 
                                     var_name='Model', value_name='Prediction')
                df_melt = df_melt.dropna(subset=['Prediction'])
                df_gp_wide = df_melt.pivot_table(index=['Group', 'Date', 'Actual'], columns='Model', values='Prediction').reset_index()
                group_list = list(df_gp_wide['Group'].unique())
                selected_group = st.selectbox("Select group", group_list)
                df_gp_sel = df_gp_wide[df_gp_wide['Group'] == selected_group]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_gp_sel['Date'], y=df_gp_sel['Actual'], mode='lines+markers', name='Actual'))
                for model_name in selected_models:
                    if model_name in df_gp_sel.columns:
                        fig2.add_trace(go.Scatter(x=df_gp_sel['Date'], y=df_gp_sel[model_name], mode='lines+markers', name=model_name))
                st.plotly_chart(fig2, use_container_width=True)

        # Download results
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
