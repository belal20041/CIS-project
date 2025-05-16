import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import traceback

# Import modeling libraries with fallbacks
try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        Prophet = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError:
        ARIMA = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from sklearn.ensemble import RandomForestRegressor

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KerasLSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    tf = None
    KerasLSTM = None
    MinMaxScaler = None

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("Time Series Forecasting Dashboard")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['date_col'] = None
    st.session_state['value_col'] = None
    st.session_state['group_col'] = None
    st.session_state['models'] = None
    st.session_state['mode'] = None

def create_supervised(df, date_col, value_col, n_lags):
    """
    Create supervised learning dataset with lag features.
    """
    data = df.copy()
    data = data.sort_values(date_col)
    for lag in range(1, n_lags+1):
        data[f"lag_{lag}"] = data[value_col].shift(lag)
    data = data.dropna().reset_index(drop=True)
    return data

def train_prophet(df, date_col, value_col, periods, seasonality_mode):
    """
    Train a Prophet model.
    """
    if Prophet is None:
        raise ImportError("Prophet library is not installed.")
    data = df.rename(columns={date_col: 'ds', value_col: 'y'})[['ds','y']].dropna()
    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(data)
    return model

def train_arima(df, date_col, value_col, order):
    """
    Train an ARIMA model.
    """
    if ARIMA is None:
        raise ImportError("statsmodels ARIMA is not available.")
    y = df.sort_values(date_col)[value_col].astype(float)
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    return model_fit

def train_sarima(df, date_col, value_col, order, seasonal_order):
    """
    Train a SARIMAX (seasonal ARIMA) model.
    """
    if SARIMAX is None:
        raise ImportError("statsmodels SARIMAX is not available.")
    y = df.sort_values(date_col)[value_col].astype(float)
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def train_lightgbm(df, date_col, value_col, params):
    """
    Train a LightGBM regressor with lag features.
    """
    if lgb is None:
        raise ImportError("LightGBM is not installed.")
    data = df.sort_values(date_col)
    n_lags = params.get('lag', 1)
    sup = create_supervised(data, date_col, value_col, n_lags)
    if sup.empty:
        raise ValueError("Not enough data for creating lag features.")
    X = sup[[f'lag_{i}' for i in range(1, n_lags+1)]].values
    y = sup[value_col].values
    lgb_params = {
        "n_estimators": params.get('n_estimators', 100),
        "learning_rate": params.get('learning_rate', 0.1),
        "max_depth": params.get('max_depth', None)
    }
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y)
    return model

def train_xgboost(df, date_col, value_col, params):
    """
    Train an XGBoost regressor with lag features.
    """
    if xgb is None:
        raise ImportError("XGBoost is not installed.")
    data = df.sort_values(date_col)
    n_lags = params.get('lag', 1)
    sup = create_supervised(data, date_col, value_col, n_lags)
    if sup.empty:
        raise ValueError("Not enough data for creating lag features.")
    X = sup[[f'lag_{i}' for i in range(1, n_lags+1)]].values
    y = sup[value_col].values
    xgb_params = {
        "n_estimators": params.get('n_estimators', 100),
        "learning_rate": params.get('learning_rate', 0.1),
        "max_depth": params.get('max_depth', None)
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)
    return model

def train_random_forest(df, date_col, value_col, params):
    """
    Train a RandomForest regressor with lag features.
    """
    data = df.sort_values(date_col)
    n_lags = params.get('lag', 1)
    sup = create_supervised(data, date_col, value_col, n_lags)
    if sup.empty:
        raise ValueError("Not enough data for creating lag features.")
    X = sup[[f'lag_{i}' for i in range(1, n_lags+1)]].values
    y = sup[value_col].values
    rf_params = {
        "n_estimators": params.get('n_estimators', 100),
        "max_depth": params.get('max_depth', None)
    }
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)
    return model

def train_lstm(df, date_col, value_col, params):
    """
    Train an LSTM model.
    """
    if tf is None or KerasLSTM is None:
        raise ImportError("TensorFlow/Keras is not installed.")
    data = df.sort_values(date_col)
    series = data[value_col].astype(float).values
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    seq_len = params.get('seq_len', 10)
    X, y = [], []
    for i in range(len(series_scaled) - seq_len):
        X.append(series_scaled[i:i+seq_len, 0])
        y.append(series_scaled[i+seq_len, 0])
    X = np.array(X); y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(KerasLSTM(50, activation='relu', input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 16), verbose=0)
    return {"model": model, "scaler": scaler, "seq_len": seq_len}

# UI Layout
tab1, tab2, tab3 = st.tabs(["Data Upload", "Model Training", "Prediction"])

with tab1:
    st.header("Data Upload")
    st.markdown("Upload your time series CSV data with date and target columns.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None
        if df is not None:
            st.write("Preview of data:")
            st.dataframe(df.head())
            date_col = st.selectbox("Select date column", df.columns)
            value_col = st.selectbox("Select target column", df.columns)
            group_col = st.selectbox("Select grouping column (optional)", [None] + list(df.columns))
            if date_col and value_col:
                if date_col == value_col:
                    st.error("Date column and target column must be different.")
                else:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.session_state['df'] = df
                        st.session_state['date_col'] = date_col
                        st.session_state['value_col'] = value_col
                        st.session_state['group_col'] = group_col
                        st.success("Data successfully loaded and parsed!")
                    except Exception as e:
                        st.error(f"Error parsing date column: {e}")

with tab2:
    st.header("Model Training")
    if st.session_state['df'] is None:
        st.info("Please upload data in the Data Upload tab first.")
        st.stop()
    df = st.session_state['df']
    date_col = st.session_state['date_col']
    value_col = st.session_state['value_col']
    group_col = st.session_state['group_col']

    st.write("Data summary:")
    st.write(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
    if group_col:
        st.write(f"Grouping by: {group_col}")
    else:
        st.write("No grouping (global model).")

    # Date range for training data
    df_sorted = df.sort_values(date_col)
    min_date = df_sorted[date_col].min().date()
    max_date = df_sorted[date_col].max().date()
    start_date = st.date_input("Start date for training data", value=min_date)
    end_date = st.date_input("End date for training data", value=max_date)
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
        st.stop()
    df_train = df_sorted[(df_sorted[date_col] >= pd.to_datetime(start_date)) &
                         (df_sorted[date_col] <= pd.to_datetime(end_date))]
    if df_train.empty:
        st.error("No data in the selected date range.")
        st.stop()

    mode = st.radio("Training mode", ["Global", "Group"])
    st.session_state['mode'] = mode
    groups = None
    if mode == "Group":
        if not group_col:
            group_col = st.selectbox("Select grouping column", 
                                     [c for c in df_train.columns if c not in [date_col, value_col]])
            st.session_state['group_col'] = group_col
        if group_col:
            groups = df_train[group_col].unique().tolist()
            st.write(f"Groups: {groups}")
        else:
            st.error("Grouping mode selected but no group column provided.")
            st.stop()

    models_selected = st.multiselect("Select model(s) to train",
                                     ["Prophet", "ARIMA", "SARIMA", "LightGBM", "XGBoost", "RandomForest", "LSTM"])
    if not models_selected:
        st.warning("Please select at least one model.")
        st.stop()

    # Parameter configuration
    prophet_params = {}
    arima_params = {}
    sarima_params = {}
    lgb_params = {}
    xgb_params = {}
    rf_params = {}
    lstm_params = {}

    if "Prophet" in models_selected:
        with st.expander("Prophet Parameters"):
            st.subheader("Prophet Settings")
            periods = st.number_input("Forecast periods (for prediction)", min_value=1, value=30, step=1)
            seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"])
            prophet_params['periods'] = periods
            prophet_params['seasonality_mode'] = seasonality_mode
    if "ARIMA" in models_selected:
        with st.expander("ARIMA Parameters"):
            st.subheader("ARIMA Settings")
            p = st.number_input("p (AR order)", min_value=0, value=1, step=1)
            d = st.number_input("d (difference order)", min_value=0, value=1, step=1)
            q = st.number_input("q (MA order)", min_value=0, value=1, step=1)
            arima_params['order'] = (p, d, q)
    if "SARIMA" in models_selected:
        with st.expander("SARIMA Parameters"):
            st.subheader("SARIMA Settings")
            p = st.number_input("p (AR order)", min_value=0, value=1, step=1)
            d = st.number_input("d (difference order)", min_value=0, value=1, step=1)
            q = st.number_input("q (MA order)", min_value=0, value=1, step=1)
            P = st.number_input("P (seasonal AR order)", min_value=0, value=1, step=1)
            D = st.number_input("D (seasonal diff order)", min_value=0, value=1, step=1)
            Q = st.number_input("Q (seasonal MA order)", min_value=0, value=1, step=1)
            m = st.number_input("m (seasonal period)", min_value=1, value=12, step=1)
            sarima_params['order'] = (p, d, q)
            sarima_params['seasonal_order'] = (P, D, Q, m)
    if "LightGBM" in models_selected:
        with st.expander("LightGBM Parameters"):
            st.subheader("LightGBM Settings")
            lgb_lag = st.number_input("Lag features", min_value=1, value=1, step=1)
            n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1)
            learning_rate = st.number_input("learning_rate", min_value=0.01, value=0.1, step=0.01, format="%.2f")
            max_depth = st.number_input("max_depth (0 = no limit)", min_value=0, value=0, step=1)
            lgb_params['lag'] = int(lgb_lag)
            lgb_params['n_estimators'] = int(n_estimators)
            lgb_params['learning_rate'] = float(learning_rate)
            lgb_params['max_depth'] = int(max_depth) if max_depth != 0 else None
    if "XGBoost" in models_selected:
        with st.expander("XGBoost Parameters"):
            st.subheader("XGBoost Settings")
            xgb_lag = st.number_input("Lag features", min_value=1, value=1, step=1, key="xgb_lag")
            n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1, key="xgb_n")
            learning_rate = st.number_input("learning_rate", min_value=0.01, value=0.1, step=0.01, format="%.2f", key="xgb_lr")
            max_depth = st.number_input("max_depth (0 = no limit)", min_value=0, value=0, step=1, key="xgb_depth")
            xgb_params['lag'] = int(xgb_lag)
            xgb_params['n_estimators'] = int(n_estimators)
            xgb_params['learning_rate'] = float(learning_rate)
            xgb_params['max_depth'] = int(max_depth) if max_depth != 0 else None
    if "RandomForest" in models_selected:
        with st.expander("RandomForest Parameters"):
            st.subheader("RandomForest Settings")
            rf_lag = st.number_input("Lag features", min_value=1, value=1, step=1, key="rf_lag")
            n_estimators = st.number_input("n_estimators", min_value=1, value=100, step=1, key="rf_n")
            max_depth = st.number_input("max_depth (0 = no limit)", min_value=0, value=0, step=1, key="rf_depth")
            rf_params['lag'] = int(rf_lag)
            rf_params['n_estimators'] = int(n_estimators)
            rf_params['max_depth'] = int(max_depth) if max_depth != 0 else None
    if "LSTM" in models_selected:
        with st.expander("LSTM Parameters"):
            st.subheader("LSTM Settings")
            seq_len = st.number_input("Sequence length", min_value=1, value=10, step=1)
            epochs = st.number_input("Epochs", min_value=1, value=10, step=1)
            batch_size = st.number_input("Batch size", min_value=1, value=16, step=1)
            lstm_params['seq_len'] = int(seq_len)
            lstm_params['epochs'] = int(epochs)
            lstm_params['batch_size'] = int(batch_size)

    if st.button("Train Models"):
        models_dict = {}
        error_models = []
        total = len(models_selected)
        progress_bar = st.progress(0)
        for i, model_name in enumerate(models_selected):
            st.write(f"Training {model_name}...")
            try:
                if mode == "Global":
                    if model_name == "Prophet":
                        model_obj = train_prophet(df_train, date_col, value_col,
                                                  prophet_params.get('periods', 30),
                                                  prophet_params.get('seasonality_mode', 'additive'))
                    elif model_name == "ARIMA":
                        model_obj = train_arima(df_train, date_col, value_col,
                                                arima_params.get('order', (1,1,1)))
                    elif model_name == "SARIMA":
                        model_obj = train_sarima(df_train, date_col, value_col,
                                                 sarima_params.get('order', (1,1,1)),
                                                 sarima_params.get('seasonal_order', (1,1,1,12)))
                    elif model_name == "LightGBM":
                        model_obj = train_lightgbm(df_train, date_col, value_col, lgb_params)
                    elif model_name == "XGBoost":
                        model_obj = train_xgboost(df_train, date_col, value_col, xgb_params)
                    elif model_name == "RandomForest":
                        model_obj = train_random_forest(df_train, date_col, value_col, rf_params)
                    elif model_name == "LSTM":
                        model_obj = train_lstm(df_train, date_col, value_col, lstm_params)
                    else:
                        model_obj = None
                    models_dict[model_name] = model_obj
                else:
                    group_models = {}
                    for grp in groups:
                        df_grp = df_train[df_train[group_col] == grp]
                        if model_name == "Prophet":
                            model_grp = train_prophet(df_grp, date_col, value_col,
                                                      prophet_params.get('periods', 30),
                                                      prophet_params.get('seasonality_mode', 'additive'))
                        elif model_name == "ARIMA":
                            model_grp = train_arima(df_grp, date_col, value_col,
                                                    arima_params.get('order', (1,1,1)))
                        elif model_name == "SARIMA":
                            model_grp = train_sarima(df_grp, date_col, value_col,
                                                     sarima_params.get('order', (1,1,1)),
                                                     sarima_params.get('seasonal_order', (1,1,1,12)))
                        elif model_name == "LightGBM":
                            model_grp = train_lightgbm(df_grp, date_col, value_col, lgb_params)
                        elif model_name == "XGBoost":
                            model_grp = train_xgboost(df_grp, date_col, value_col, xgb_params)
                        elif model_name == "RandomForest":
                            model_grp = train_random_forest(df_grp, date_col, value_col, rf_params)
                        elif model_name == "LSTM":
                            model_grp = train_lstm(df_grp, date_col, value_col, lstm_params)
                        else:
                            model_grp = None
                        group_models[grp] = model_grp
                    models_dict[model_name] = group_models
                st.success(f"{model_name} trained successfully.")
            except Exception as e:
                error_models.append((model_name, str(e)))
                st.error(f"Error training {model_name}: {e}")
                st.error(traceback.format_exc())
            progress_bar.progress(int((i+1)/total * 100))
        st.session_state['models'] = models_dict
        st.session_state['model_params'] = {
            "Prophet": prophet_params,
            "ARIMA": arima_params,
            "SARIMA": sarima_params,
            "LightGBM": lgb_params,
            "XGBoost": xgb_params,
            "RandomForest": rf_params,
            "LSTM": lstm_params
        }
        if error_models:
            failed = ", ".join([m for m,_ in error_models])
            st.warning(f"Models failed to train: {failed}")
        else:
            st.success("All models trained successfully!")

with tab3:
    st.header("Prediction")
    if st.session_state['models'] is None:
        st.info("Please train models in the Model Training tab first.")
        st.stop()
    models_dict = st.session_state['models']
    mode = st.session_state.get('mode', 'Global')
    df = st.session_state['df']
    date_col = st.session_state['date_col']
    value_col = st.session_state['value_col']
    group_col = st.session_state['group_col']

    model_choice = st.selectbox("Select model for prediction", list(models_dict.keys()))
    model_obj = models_dict[model_choice]
    if isinstance(model_obj, dict):
        group_choice = st.selectbox("Select group for prediction", list(model_obj.keys()))
        model_obj = model_obj[group_choice]
        df_input = df[df[group_col] == group_choice].copy()
    else:
        df_input = df.copy()

    forecast_periods = st.number_input("Forecast periods", min_value=1, value=30, step=1)
    if st.button("Generate Forecast"):
        try:
            if model_choice == "Prophet":
                future = model_obj.make_future_dataframe(periods=forecast_periods)
                forecast = model_obj.predict(future)
                st.write("Forecast results:")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods))
                st.line_chart(forecast.set_index('ds')['yhat'])
            elif model_choice in ["ARIMA", "SARIMA"]:
                fcst = model_obj.get_forecast(steps=forecast_periods)
                summary = fcst.summary_frame()
                st.write("Forecast results:")
                st.dataframe(summary[['mean', 'mean_ci_lower', 'mean_ci_upper']])
                st.line_chart(summary['mean'])
            elif model_choice in ["LightGBM", "XGBoost", "RandomForest"]:
                params = st.session_state.get('model_params', {})
                if model_choice == "LightGBM":
                    lag = params.get('LightGBM', {}).get('lag', 1)
                elif model_choice == "XGBoost":
                    lag = params.get('XGBoost', {}).get('lag', 1)
                else:
                    lag = params.get('RandomForest', {}).get('lag', 1)
                series = df_input.sort_values(date_col)[value_col].tolist()
                forecasts = []
                for _ in range(forecast_periods):
                    if len(series) < lag:
                        st.error("Not enough data to make lag-based forecasts.")
                        break
                    X_input = np.array(series[-lag:]).reshape(1, -1)
                    y_pred = model_obj.predict(X_input)[0]
                    forecasts.append(y_pred)
                    series.append(y_pred)
                last_dt = pd.to_datetime(df_input[date_col].max())
                try:
                    freq = pd.infer_freq(df_input[date_col])
                    if freq:
                        future_dates = pd.date_range(start=last_dt + pd.Timedelta(1, unit=freq), periods=len(forecasts), freq=freq)
                    else:
                        raise ValueError
                except Exception:
                    future_dates = [last_dt + pd.Timedelta(days=i) for i in range(1, len(forecasts)+1)]
                result = pd.DataFrame({"ds": future_dates, "prediction": forecasts})
                st.write("Forecast results:")
                st.dataframe(result)
                st.line_chart(result.set_index('ds')['prediction'])
            elif model_choice == "LSTM":
                model_map = model_obj
                net = model_map['model']
                scaler = model_map['scaler']
                seq_len = model_map['seq_len']
                series = df_input.sort_values(date_col)[value_col].values.astype(float)
                scaled_series = scaler.transform(series.reshape(-1,1)).flatten().tolist()
                forecasts = []
                for _ in range(forecast_periods):
                    if len(scaled_series) < seq_len:
                        st.error("Not enough data to forecast with LSTM.")
                        break
                    X_input = np.array(scaled_series[-seq_len:]).reshape(1, seq_len, 1)
                    yhat_scaled = net.predict(X_input, verbose=0)[0][0]
                    yhat = scaler.inverse_transform([[yhat_scaled]])[0][0]
                    forecasts.append(yhat)
                    scaled_series.append(yhat_scaled)
                last_dt = pd.to_datetime(df_input[date_col].max())
                future_dates = [last_dt + pd.Timedelta(days=i) for i in range(1, len(forecasts)+1)]
                result = pd.DataFrame({"ds": future_dates, "prediction": forecasts})
                st.write("Forecast results:")
                st.dataframe(result)
                st.line_chart(result.set_index('ds')['prediction'])
            else:
                st.error("Selected model not recognized.")
        except Exception as e:
            st.error(f"Error during forecasting: {e}")
            st.error(traceback.format_exc())
