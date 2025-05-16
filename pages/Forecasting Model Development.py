import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Machine learning models and tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        Prophet = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    tf = None

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Streamlit app configuration
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}

st.title("Sales Forecasting App")

# Create tabs for the app
tabs = st.tabs(["Data Upload", "Model Training", "Prediction"])

# Tab 1: Data Upload
with tabs[0]:
    st.header("Upload CSV Data")
    uploaded_file = st.file_uploader("Upload your sales CSV file", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            # Reset any previously trained models
            st.session_state['models'] = {}
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.stop()
    else:
        df = None

    if st.session_state.get('data') is not None:
        df = st.session_state['data']
        st.subheader("Data Preview")
        st.write(df.head(10))
        st.write("Data summary:")
        st.write(df.describe(include='all'))
        st.write("Missing values per column:")
        st.write(df.isnull().sum())
        required_cols = {'date','family','store_nbr','sales'}
        if not required_cols.issubset(df.columns):
            st.error(f"Missing required columns: {required_cols - set(df.columns)}")
            st.stop()
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"Error parsing 'date' column: {e}")
            st.stop()
        st.success("Data is ready for modeling.")

# Tab 2: Model Training
with tabs[1]:
    st.header("Train Forecasting Models")
    if st.session_state.get('data') is None:
        st.error("Please upload data in the 'Data Upload' tab before training models.")
    else:
        data = st.session_state['data']
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                data['date'] = pd.to_datetime(data['date'])
            except Exception as e:
                st.error(f"Error converting 'date' column: {e}")
        # Model selection
        model_options = ["Prophet","LightGBM","XGBoost","RandomForest","ARIMA","SARIMA","LSTM"]
        selected_models = st.multiselect("Select models to train:", model_options)
        st.write("Selected models:", selected_models)
        global_opt = st.checkbox("Train Global Model (aggregated)", value=True)
        group_opt = st.checkbox("Train Per-Group Models (by family & store)", value=True)

        # Hyperparameter inputs
        if "ARIMA" in selected_models or "SARIMA" in selected_models:
            with st.expander("ARIMA/SARIMA Parameters"):
                arima_p = st.number_input("ARIMA p (lags)", min_value=0, max_value=10, value=1, step=1)
                arima_d = st.number_input("ARIMA d (differences)", min_value=0, max_value=2, value=0, step=1)
                arima_q = st.number_input("ARIMA q (MA order)", min_value=0, max_value=10, value=0, step=1)
                sarima_P = st.number_input("SARIMA P (seasonal lags)", min_value=0, max_value=5, value=0, step=1)
                sarima_D = st.number_input("SARIMA D (seasonal diff)", min_value=0, max_value=2, value=0, step=1)
                sarima_Q = st.number_input("SARIMA Q (seasonal MA)", min_value=0, max_value=5, value=0, step=1)
                sarima_m = st.number_input("SARIMA m (seasonal period)", min_value=0, max_value=12, value=0, step=1,
                                         help="0 or 1 for no seasonality (monthly=12, weekly=7, etc).")
        else:
            arima_p=arima_d=arima_q=sarima_P=sarima_D=sarima_Q=sarima_m=None

        if "LightGBM" in selected_models:
            with st.expander("LightGBM Parameters"):
                lgb_n_estimators = st.number_input("LightGBM n_estimators", min_value=1, value=100, step=10)
                lgb_max_depth = st.number_input("LightGBM max_depth (0 for none)", min_value=0, value=0, step=1)
                lgb_lr = st.number_input("LightGBM learning_rate", value=0.1, step=0.01, format="%.2f")
        else:
            lgb_n_estimators=lgb_max_depth=lgb_lr=None

        if "XGBoost" in selected_models:
            with st.expander("XGBoost Parameters"):
                xgb_n_estimators = st.number_input("XGBoost n_estimators", min_value=1, value=100, step=10)
                xgb_max_depth = st.number_input("XGBoost max_depth (0 for none)", min_value=0, value=0, step=1)
                xgb_lr = st.number_input("XGBoost learning_rate", value=0.1, step=0.01, format="%.2f")
        else:
            xgb_n_estimators=xgb_max_depth=xgb_lr=None

        if "RandomForest" in selected_models:
            with st.expander("RandomForest Parameters"):
                rf_n_estimators = st.number_input("RandomForest n_estimators", min_value=1, value=100, step=10)
                rf_max_depth = st.number_input("RandomForest max_depth (0 for none)", min_value=0, value=0, step=1)
        else:
            rf_n_estimators=rf_max_depth=None

        if "LSTM" in selected_models:
            with st.expander("LSTM Parameters"):
                lstm_epochs = st.number_input("LSTM epochs", min_value=1, value=5, step=1)
                lstm_lag = st.number_input("LSTM lag (days)", min_value=1, value=7, step=1)
                lstm_units = st.number_input("LSTM units", min_value=1, value=50, step=1)
        else:
            lstm_epochs=lstm_lag=lstm_units=None

        # Define training helper functions
        def train_prophet(df_train):
            model = Prophet()
            try:
                model.fit(df_train.rename(columns={'date':'ds','sales':'y'}))
            except Exception as e:
                st.error(f"Prophet training error: {e}")
                return None
            return model

        def train_lgbm_model(df_train, n_estimators, max_depth, learning_rate):
            X = df_train[['time_idx','month','day_of_week']]
            y = df_train['sales']
            model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                      max_depth=(None if max_depth==0 else max_depth))
            try:
                model.fit(X, y)
            except Exception as e:
                st.error(f"LightGBM training error: {e}")
                return None
            return model

        def train_xgb_model(df_train, n_estimators, max_depth, learning_rate):
            X = df_train[['time_idx','month','day_of_week']]
            y = df_train['sales']
            model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=(None if max_depth==0 else max_depth))
            try:
                model.fit(X, y, verbose=False)
            except Exception as e:
                st.error(f"XGBoost training error: {e}")
                return None
            return model

        def train_rf_model(df_train, n_estimators, max_depth):
            X = df_train[['time_idx','month','day_of_week']]
            y = df_train['sales']
            model = RandomForestRegressor(n_estimators=n_estimators,
                                          max_depth=(None if max_depth==0 else max_depth))
            try:
                model.fit(X, y)
            except Exception as e:
                st.error(f"RandomForest training error: {e}")
                return None
            return model

        def train_arima_model(series, order):
            try:
                model = ARIMA(series, order=order)
                model_fit = model.fit()
            except Exception as e:
                st.error(f"ARIMA training error: {e}")
                return None
            return model_fit

        def train_sarima_model(series, order, seasonal_order):
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
            except Exception as e:
                st.error(f"SARIMA training error: {e}")
                return None
            return model_fit

        def train_lstm_model(df_train, lag, epochs, units):
            series = df_train['sales'].values
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(series.reshape(-1,1))
            X, y = [], []
            for i in range(len(scaled)-lag):
                X.append(scaled[i:i+lag])
                y.append(scaled[i+lag])
            X, y = np.array(X), np.array(y)
            if X.size == 0:
                st.error("Not enough data to train LSTM with given lag.")
                return None
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = Sequential()
            model.add(LSTM(int(units), input_shape=(lag,1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            try:
                model.fit(X, y, epochs=int(epochs), verbose=0)
            except Exception as e:
                st.error(f"LSTM training error: {e}")
                return None
            return model, scaler

        # Button to start training
        if st.button("Train Models"):
            data = st.session_state['data']
            # Initialize model dicts if needed
            for m in selected_models:
                if m not in st.session_state['models']:
                    st.session_state['models'][m] = {}

            # Determine group combinations
            group_keys = []
            if group_opt:
                if 'family' not in data.columns or 'store_nbr' not in data.columns:
                    st.error("Data must have 'family' and 'store_nbr' for group training.")
                else:
                    group_keys = [(fam,store) for fam, store in data.groupby(['family','store_nbr']).groups.keys()]

            # Train each selected model
            for model_name in selected_models:
                st.write(f"Training model: {model_name}")
                st.session_state['models'].setdefault(model_name, {})

                # Global training (aggregate)
                if global_opt:
                    df_global = data.groupby('date').agg({'sales':'sum'}).reset_index()
                    df_global = df_global.sort_values('date')
                    df_global['time_idx'] = range(len(df_global))
                    df_global['month'] = df_global['date'].dt.month
                    df_global['day_of_week'] = df_global['date'].dt.dayofweek
                    if model_name == "Prophet" and Prophet is not None:
                        m = train_prophet(df_global)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "LightGBM" and lgb is not None:
                        m = train_lgbm_model(df_global, lgb_n_estimators, lgb_max_depth, lgb_lr)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "XGBoost" and xgb is not None:
                        m = train_xgb_model(df_global, xgb_n_estimators, xgb_max_depth, xgb_lr)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "RandomForest":
                        m = train_rf_model(df_global, rf_n_estimators, rf_max_depth)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "ARIMA":
                        series = df_global['sales']
                        order = (int(arima_p), int(arima_d), int(arima_q))
                        m = train_arima_model(series, order)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "SARIMA":
                        series = df_global['sales']
                        order = (int(arima_p), int(arima_d), int(arima_q))
                        seasonal = (int(sarima_P), int(sarima_D), int(sarima_Q), int(sarima_m)) if sarima_m and sarima_m>1 else (0,0,0,0)
                        m = train_sarima_model(series, order, seasonal)
                        if m: st.session_state['models'][model_name][(None,None)] = m
                    elif model_name == "LSTM" and tf is not None:
                        m = train_lstm_model(df_global, int(lstm_lag), int(lstm_epochs), int(lstm_units))
                        if m:
                            model_obj, scaler = m
                            st.session_state['models'][model_name][(None,None)] = (model_obj, scaler, int(lstm_lag))

                # Per-group training
                if group_opt and group_keys:
                    for (family, store) in group_keys:
                        df_group = data[(data['family']==family) & (data['store_nbr']==store)].copy()
                        df_group = df_group.sort_values('date')
                        if df_group.empty:
                            continue
                        df_group['time_idx'] = range(len(df_group))
                        df_group['month'] = df_group['date'].dt.month
                        df_group['day_of_week'] = df_group['date'].dt.dayofweek
                        if model_name == "Prophet" and Prophet is not None:
                            m = train_prophet(df_group)
                            if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "LightGBM" and lgb is not None:
                            m = train_lgbm_model(df_group, lgb_n_estimators, lgb_max_depth, lgb_lr)
                            if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "XGBoost" and xgb is not None:
                            m = train_xgb_model(df_group, xgb_n_estimators, xgb_max_depth, xgb_lr)
                            if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "RandomForest":
                            m = train_rf_model(df_group, rf_n_estimators, rf_max_depth)
                            if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "ARIMA":
                            series = df_group['sales']
                            if len(series) < max(arima_p, arima_d, arima_q)+1:
                                st.warning(f"Skipping ARIMA for group {(family,store)} (not enough data).")
                            else:
                                order = (int(arima_p), int(arima_d), int(arima_q))
                                m = train_arima_model(series, order)
                                if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "SARIMA":
                            series = df_group['sales']
                            req = max(arima_p, arima_d, arima_q, sarima_P, sarima_D, sarima_Q)
                            if len(series) < req+1:
                                st.warning(f"Skipping SARIMA for group {(family,store)} (not enough data).")
                            else:
                                order = (int(arima_p), int(arima_d), int(arima_q))
                                seasonal = (int(sarima_P), int(sarima_D), int(sarima_Q), int(sarima_m)) if sarima_m and sarima_m>1 else (0,0,0,0)
                                m = train_sarima_model(series, order, seasonal)
                                if m: st.session_state['models'][model_name][(family, store)] = m
                        elif model_name == "LSTM" and tf is not None:
                            if len(df_group) <= int(lstm_lag):
                                st.warning(f"Skipping LSTM for {(family,store)} (lag too large).")
                            else:
                                res = train_lstm_model(df_group, int(lstm_lag), int(lstm_epochs), int(lstm_units))
                                if res:
                                    model_obj, scaler = res
                                    st.session_state['models'][model_name][(family, store)] = (model_obj, scaler, int(lstm_lag))
            st.success("Model training completed.")

# Tab 3: Prediction
with tabs[2]:
    st.header("Make Predictions")
    if st.session_state.get('data') is None or not any(st.session_state['models'].values()):
        st.error("No data or trained models found. Please upload data and train models first.")
    else:
        data = st.session_state['data']
        model_names = [m for m in st.session_state['models'] if st.session_state['models'][m]]
        if not model_names:
            st.error("No trained models available.")
        else:
            selected_model = st.selectbox("Select model for prediction", model_names)
            combos = list(st.session_state['models'][selected_model].keys())
            families = sorted({fam for (fam,store) in combos if fam is not None})
            has_global = (None,None) in combos
            if has_global:
                families = ['All families'] + families
            family = st.selectbox("Select Product Family", families)
            if family != 'All families':
                stores = sorted({store for (fam,store) in combos if fam == family})
                store = st.selectbox("Select Store Number", stores)
            else:
                store = 'All stores'
            # Determine last date of data
            if family == 'All families' and store == 'All stores':
                last_date = data['date'].max()
            else:
                last_date = data[(data['family']==family) & (data['store_nbr']==store)]['date'].max()
            start_date = st.date_input("Forecast Start Date", value=(last_date + timedelta(days=1)))
            end_date = st.date_input("Forecast End Date", value=(last_date + timedelta(days=30)))
            if start_date <= last_date:
                st.error(f"Start date must be after last data date: {last_date.date()}")
            elif end_date <= start_date:
                st.error("End date must be after start date.")
            else:
                if st.button("Generate Forecast"):
                    horizon_all = (end_date - last_date).days
                    offset_start = (start_date - last_date).days - 1
                    if offset_start < 0:
                        offset_start = 0
                    # Prepare historical data for chart
                    if family == 'All families' and store == 'All stores':
                        df_group = data.groupby('date').agg({'sales':'sum'}).reset_index()
                    else:
                        df_group = data[(data['family']==family) & (data['store_nbr']==store)].copy()
                    df_group = df_group.sort_values('date')
                    # Retrieve model object
                    key = (None,None) if (family=='All families' and store=='All stores') else (family, store)
                    model_obj = st.session_state['models'][selected_model].get(key)
                    if model_obj is None:
                        st.error("No trained model for this selection.")
                    else:
                        forecast_dates = pd.date_range(start=last_date+timedelta(days=1), periods=horizon_all)
                        # Prediction logic
                        if selected_model == "Prophet":
                            m = model_obj
                            future_df = m.make_future_dataframe(periods=horizon_all, include_history=False)
                            forecast = m.predict(future_df)[['ds','yhat']].rename(columns={'ds':'date','yhat':'forecast'})
                            forecast = forecast[forecast['date'] >= pd.to_datetime(start_date)].reset_index(drop=True)
                        elif selected_model == "LightGBM":
                            model = model_obj
                            X_future = pd.DataFrame({
                                'time_idx': range(len(df_group), len(df_group)+horizon_all),
                                'month': [d.month for d in forecast_dates],
                                'day_of_week': [d.dayofweek for d in forecast_dates]
                            })
                            preds = model.predict(X_future)
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        elif selected_model == "XGBoost":
                            model = model_obj
                            X_future = pd.DataFrame({
                                'time_idx': range(len(df_group), len(df_group)+horizon_all),
                                'month': [d.month for d in forecast_dates],
                                'day_of_week': [d.dayofweek for d in forecast_dates]
                            })
                            preds = model.predict(X_future)
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        elif selected_model == "RandomForest":
                            model = model_obj
                            X_future = pd.DataFrame({
                                'time_idx': range(len(df_group), len(df_group)+horizon_all),
                                'month': [d.month for d in forecast_dates],
                                'day_of_week': [d.dayofweek for d in forecast_dates]
                            })
                            preds = model.predict(X_future)
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        elif selected_model == "ARIMA":
                            model_fit = model_obj
                            preds = model_fit.forecast(steps=horizon_all)
                            preds = np.array(preds)
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        elif selected_model == "SARIMA":
                            model_fit = model_obj
                            try:
                                pred_res = model_fit.get_forecast(steps=horizon_all)
                                preds = pred_res.predicted_mean.values
                            except:
                                preds = np.array(model_fit.forecast(steps=horizon_all))
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        elif selected_model == "LSTM":
                            model, scaler, lag = model_obj
                            values = df_group['sales'].values
                            scaled = scaler.transform(values.reshape(-1,1)).flatten()
                            last_window = list(scaled[-lag:])
                            preds_scaled = []
                            for _ in range(horizon_all):
                                x_input = np.array(last_window[-lag:]).reshape((1, lag, 1))
                                yhat = model.predict(x_input, verbose=0)[0][0]
                                preds_scaled.append(yhat)
                                last_window.append(yhat)
                            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
                            if offset_start > 0:
                                preds = preds[offset_start:]
                                forecast_dates = pd.date_range(start=start_date, end=end_date)
                            forecast = pd.DataFrame({'date': forecast_dates, 'forecast': preds})
                        else:
                            st.error("Selected model cannot be used for prediction.")
                            forecast = pd.DataFrame()

                        # Display results
                        if not forecast.empty:
                            st.subheader("Forecast Results")
                            st.write(forecast)
                            # Plot actual vs forecast
                            if not df_group.empty:
                                chart_df = pd.DataFrame({
                                    'date': list(df_group['date']) + list(forecast['date']),
                                    'Sales': list(df_group['sales']) + [None]*len(forecast),
                                    'Forecast': [None]*len(df_group) + list(forecast['forecast'])
                                }).set_index('date')
                                st.line_chart(chart_df)
                            else:
                                st.line_chart(forecast.rename(columns={'forecast':'Sales'}).set_index('date'))
