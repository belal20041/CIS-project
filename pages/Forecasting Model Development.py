import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import psutil
import warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Sales Forecasting Dashboard")

# Initialize session state to store results
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'train_set' not in st.session_state:
    st.session_state.train_set = None
if 'val_set' not in st.session_state:
    st.session_state.val_set = None
if 'test' not in st.session_state:
    st.session_state.test = None
if 'sub' not in st.session_state:
    st.session_state.sub = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# Tabs
training_tab, prediction_tab = st.tabs(["Training", "Prediction"])

# Constants
TRAIN_END = '2017-07-15'
VAL_END = '2017-08-15'

# Cache data loading and processing
@st.cache_data
def load_and_process_data(train_file, test_file, sub_file):
    try:
        # Load data
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        sub = pd.read_csv(sub_file)
        
        # Data preprocessing (from original Cells 3–6)
        train['date'] = pd.to_datetime(train['date'])
        test['date'] = pd.to_datetime(test['date'], format='%d-%m-%Y')
        train[['store_nbr', 'onpromotion']] = train[['store_nbr', 'onpromotion']].astype('int32')
        test[['store_nbr', 'onpromotion']] = test[['store_nbr', 'onpromotion']].astype('int32')
        train['sales'] = train['sales'].astype('float32')
        train.dropna(subset=['date'], inplace=True)
        test.dropna(subset=['date'], inplace=True)
        
        # Prepare data
        train['is_train'] = 1
        test['is_train'] = 0
        combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', 'date'])
        agg_dict = {'sales': 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
        combined = combined.groupby(['store_nbr', 'family', 'date']).agg(agg_dict).reset_index()
        combined = combined.astype({'store_nbr': 'int32', 'family': 'category', 'date': 'datetime64[ns]', 
                                   'sales': 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
        
        # Handle missing values
        grouped = combined.groupby(['store_nbr', 'family'])
        processed_groups = []
        for (store_nbr, family), group in grouped:
            group['sales'] = group['sales'].ffill().fillna(0).astype('float32')
            group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
            processed_groups.append(group)
        combined = pd.concat(processed_groups)
        
        # Add features
        combined['day'] = combined['date'].dt.day.astype('int8')
        combined['dow'] = combined['date'].dt.dayofweek.astype('int8')
        combined['month'] = combined['date'].dt.month.astype('int8')
        combined['year'] = combined['date'].dt.year.astype('int16')
        combined['sin_month'] = np.sin(2 * np.pi * combined['month'] / 12).astype('float32')
        lags = [7, 14]
        for lag in lags:
            combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag).astype('float32')
        combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).mean().astype('float32')
        combined['store_nbr_encoded'] = LabelEncoder().fit_transform(combined['store_nbr']).astype('int8')
        combined['family_encoded'] = LabelEncoder().fit_transform(combined['family']).astype('int8')
        feature_cols = ['onpromotion', 'day', 'dow', 'month', 'year', 'sin_month', 'store_nbr_encoded', 
                        'family_encoded', 'lag_7', 'lag_14', 'roll_mean_7']
        combined[feature_cols] = StandardScaler().fit_transform(combined[feature_cols].fillna(0)).astype('float32')
        
        # Split data
        train = combined[combined['is_train'] == 1]
        test = combined[combined['is_train'] == 0].drop(['sales'], axis=1)
        train_set = train[train['date'] <= TRAIN_END]
        val_set = train[(train['date'] > TRAIN_END) & (train['date'] <= VAL_END)]
        
        return train_set, val_set, test, sub, feature_cols
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None, None, None, None

# Training Tab
with training_tab:
    st.header("Train Forecasting Models")
    
    # File uploaders
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")
    sub_file = st.file_uploader("Upload Submission CSV", type="csv", key="uploader_sub")
    
    # Model selection
    models = ["Naive", "Seasonal Naive", "Exponential Smoothing", "Holt's Linear Trend", 
              "Moving Average", "Linear Regression", "XGBoost", "ARIMA", "SARIMA", "Prophet", "LSTM"]
    selected_models = st.multiselect("Select Models to Train", models, default=["XGBoost"])
    
    # Train button
    train_button = st.button("Train Models")
    
    st.divider()  # Separator between inputs and outputs
    
    if train_button and train_file and test_file and sub_file and selected_models:
        with st.spinner("Training models..."):
            # Load and process data
            train_set, val_set, test, sub, feature_cols = load_and_process_data(train_file, test_file, sub_file)
            if train_set is None:
                st.error("Failed to process data. Please check the file format and try again.")
            else:
                st.session_state.train_set = train_set
                st.session_state.val_set = val_set
                st.session_state.test = test
                st.session_state.sub = sub
                st.session_state.feature_cols = feature_cols
                
                # Training logic for each model
                for model_name in selected_models:
                    st.write(f"Training {model_name}...")
                    temp_dir = tempfile.gettempdir()
                    
                    if model_name == "Naive":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        naive_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            last_value = group['sales'].iloc[-1]
                            naive_preds[(store, family)] = np.full(val_steps, last_value)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(naive_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Naive Forecast Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "naive_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in naive_preds.items()}
                        }
                    
                    elif model_name == "Seasonal Naive":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        season_length = 7  # Weekly seasonality
                        seasonal_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            last_season = group['sales'].tail(season_length).values
                            preds = np.tile(last_season, (val_steps // season_length) + 1)[:val_steps]
                            seasonal_preds[(store, family)] = preds
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(seasonal_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Seasonal Naive Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "seasonal_naive_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in seasonal_preds.items()}
                        }
                    
                    elif model_name == "Exponential Smoothing":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        es_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            model = SimpleExpSmoothing(group['sales']).fit(smoothing_level=0.3, optimized=False)
                            es_preds[(store, family)] = model.forecast(val_steps)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(es_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Exponential Smoothing Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "es_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in es_preds.items()}
                        }
                    
                    elif model_name == "Holt's Linear Trend":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        holt_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            model = Holt(group['sales']).fit(smoothing_level=0.3, smoothing_trend=0.1)
                            holt_preds[(store, family)] = model.forecast(val_steps)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(holt_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Holt's Linear Trend Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "holt_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in holt_preds.items()}
                        }
                    
                    elif model_name == "Moving Average":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        window = 7
                        ma_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            last_window = group['sales'].tail(window).mean()
                            ma_preds[(store, family)] = np.full(val_steps, last_window)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(ma_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Moving Average Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "ma_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in ma_preds.items()}
                        }
                    
                    elif model_name == "Linear Regression":
                        X_train = train_set[feature_cols]
                        y_train = train_set['sales']
                        X_val = val_set[feature_cols]
                        y_val = val_set['sales']
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        actual = np.clip(y_val, 0, None)
                        predicted = np.clip(y_pred, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_val.values[:100], label='Actual')
                        plt.plot(y_pred[:100], label='Predicted')
                        plt.title("Linear Regression Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "lr_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        # Store predictions per group for consistency
                        pred_dict = {}
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            mask = val_set.index.isin(group.index)
                            pred_dict[(store, family)] = y_pred[mask].tolist()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': y_val.values.tolist(),
                            'y_pred': pred_dict
                        }
                    
                    elif model_name == "XGBoost":
                        X_train = train_set[feature_cols]
                        y_train = train_set['sales']
                        X_val = val_set[feature_cols]
                        y_val = val_set['sales']
                        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        actual = np.clip(y_val, 0, None)
                        predicted = np.clip(y_pred, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_val.values[:100], label='Actual')
                        plt.plot(y_pred[:100], label='Predicted')
                        plt.title("XGBoost Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "xgb_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        # Store predictions per group
                        pred_dict = {}
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            mask = val_set.index.isin(group.index)
                            pred_dict[(store, family)] = y_pred[mask].tolist()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': y_val.values.tolist(),
                            'y_pred': pred_dict
                        }
                    
                    elif model_name == "ARIMA":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        arima_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            model = auto_arima(group['sales'], seasonal=False, max_p=3, max_q=3, trace=False)
                            arima_preds[(store, family)] = model.predict(val_steps)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(arima_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("ARIMA Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "arima_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in arima_preds.items()}
                        }
                    
                    elif model_name == "SARIMA":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        val_steps = len(val_dates)
                        sarima_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            model = auto_arima(group['sales'], seasonal=True, m=7, max_p=3, max_q=3, trace=False)
                            sarima_preds[(store, family)] = model.predict(val_steps)
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(sarima_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("SARIMA Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "sarima_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in sarima_preds.items()}
                        }
                    
                    elif model_name == "Prophet":
                        val_dates = pd.date_range('2017-07-16', '2017-08-15')
                        prophet_preds = {}
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            df = group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                            model.fit(df)
                            prophet_preds[(store, family)] = model.predict(pd.DataFrame({'ds': val_dates}))['yhat'].values
                        actuals = []
                        preds = []
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(prophet_preds[(store, family)])
                        actual = np.clip(actuals, 0, None)
                        predicted = np.clip(preds, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(actuals[:100], label='Actual')
                        plt.plot(preds[:100], label='Predicted')
                        plt.title("Prophet Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "prophet_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': actuals,
                            'y_pred': {k: v.tolist() for k, v in prophet_preds.items()}
                        }
                    
                    elif model_name == "LSTM":
                        seq_length = 7
                        X_train = []
                        y_train = []
                        for _, g in train_set.groupby(['store_nbr', 'family']):
                            g = g.sort_values('date')
                            for i in range(len(g) - seq_length):
                                X_train.append(g.iloc[i:i+seq_length][feature_cols].values)
                                y_train.append(g.iloc[i+seq_length]['sales'])
                        X_train = np.array(X_train)
                        y_train = np.array(y_train)
                        X_val = []
                        y_val = []
                        val_indices = []
                        for (store, family), g in val_set.groupby(['store_nbr', 'family']):
                            g = g.sort_values('date')
                            for i in range(len(g) - seq_length):
                                X_val.append(g.iloc[i:i+seq_length][feature_cols].values)
                                y_val.append(g.iloc[i+seq_length]['sales'])
                                val_indices.append((store, family, i))
                        X_val = np.array(X_val)
                        y_val = np.array(y_val)
                        model = Sequential([LSTM(50, activation='relu', input_shape=(seq_length, len(feature_cols))), Dense(1)])
                        model.compile(optimizer='adam', loss='mse')
                        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=0)
                        y_pred = model.predict(X_val, verbose=0).flatten()
                        actual = np.clip(y_val, 0, None)
                        predicted = np.clip(y_pred, 0, None)
                        metrics = {
                            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                            'mae': mean_absolute_error(actual, predicted),
                            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                        }
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_val[:100], label='Actual')
                        plt.plot(y_pred[:100], label='Predicted')
                        plt.title("LSTM Predictions")
                        plt.legend()
                        plot_path = os.path.join(temp_dir, "lstm_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        # Store predictions per group
                        pred_dict = {}
                        for (store, family, _), pred in zip(val_indices, y_pred):
                            if (store, family) not in pred_dict:
                                pred_dict[(store, family)] = []
                            pred_dict[(store, family)].append(pred)
                        st.session_state.model_results[model_name] = {
                            'metrics': metrics,
                            'plot_path': plot_path,
                            'y_val': y_val.tolist(),
                            'y_pred': {k: v for k, v in pred_dict.items()}
                        }
                    
                    # Display metrics
                    st.write(f"### {model_name} Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                    col3.metric("MAE", f"{metrics['mae']:.4f}")
                    col4.metric("MAPE", f"{metrics['mape']:.4f}")
                
                # Display memory usage
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                st.metric("Memory Usage (MB)", f"{mem:.2f}")
                st.success("Training completed!")
    elif train_button:
        st.error("Please upload all CSV files and select at least one model.")

# Prediction Tab
with prediction_tab:
    st.header("Visualize Predictions vs Actual")
    
    # Check if data is loaded
    if st.session_state.train_set is not None:
        # Get store numbers and families
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        # User inputs
        store_nbr = st.selectbox("Select Store Number", store_nbrs)
        family = st.selectbox("Select Product Family", families)
        selected_models = st.multiselect("Select Models to Visualize", models, default=["XGBoost"])
        
        st.divider()  # Separator between inputs and outputs
        
        if selected_models:
            for model_name in selected_models:
                st.subheader(f"{model_name} Predictions")
                if model_name in st.session_state.model_results:
                    result = st.session_state.model_results[model_name]
                    metrics = result['metrics']
                    plot_path = result['plot_path']
                    
                    # Filter validation set for selected store and family
                    val_set = st.session_state.val_set
                    mask = (val_set['store_nbr'] == store_nbr) & (val_set['family'] == family)
                    if mask.sum() > 0:
                        group = val_set[mask].sort_values('date')
                        actual = group['sales'].values[:100]
                        
                        # Get predictions for the selected store-family pair
                        key = (store_nbr, family)
                        pred = result['y_pred'].get(key)
                        if pred is not None and len(pred) > 0:
                            pred = np.array(pred)[:100]
                            # Plot actual vs predicted
                            plt.figure(figsize=(10, 5))
                            plt.plot(actual, label='Actual', color='blue')
                            plt.plot(pred, label='Predicted', color='orange')
                            plt.title(f"{model_name} Predictions: Store {store_nbr}, Family {family}")
                            plt.xlabel("Time")
                            plt.ylabel("Sales")
                            plt.legend()
                            plot_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_custom_pred.png")
                            plt.savefig(plot_path)
                            plt.close()
                            
                            # Display plot
                            if os.path.exists(plot_path):
                                image = Image.open(plot_path)
                                st.image(image, caption=f"{model_name} Predictions vs Actual", use_column_width=True)
                            else:
                                st.write("Plot not available.")
                            
                            # Display metrics
                            st.write("### Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                            col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                            col3.metric("MAE", f"{metrics['mae']:.4f}")
                            col4.metric("MAPE", f"{metrics['mape']:.4f}")
                        else:
                            st.write("Predictions not available for this store-family combination.")
                    else:
                        st.write("No validation data for this store-family combination.")
                else:
                    st.write("Model not trained. Please train the model in the Training tab.")
        else:
            st.write("Please select at least one model to visualize.")
    else:
        st.write("Please upload data and train models in the Training tab.")
