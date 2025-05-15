import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Set page config
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

# Initialize session state
if 'train_df' not in st.session_state:
    st.session_state['train_df'] = None
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

# Sidebar: upload data
st.sidebar.title("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Function to load CSV into session state
def load_csv(file_uploader, state_key, name_key):
    if file_uploader is not None:
        try:
            df = pd.read_csv(file_uploader)
        except Exception:
            st.sidebar.error(f"Error: Failed to read {name_key} CSV.")
            return
        # If new file or changed file, update and reset training flag
        if (st.session_state[state_key] is None 
                or (name_key in st.session_state and file_uploader.name != st.session_state[name_key])):
            st.session_state[state_key] = df
            st.session_state[name_key] = file_uploader.name
            st.session_state['trained'] = False

load_csv(train_file, 'train_df', 'train_filename')
load_csv(test_file, 'test_df', 'test_filename')

if st.session_state['train_df'] is not None:
    st.sidebar.success(f"Train data loaded: {st.session_state['train_df'].shape[0]} rows")
if st.session_state['test_df'] is not None:
    st.sidebar.success(f"Test data loaded: {st.session_state['test_df'].shape[0]} rows")

# Data preprocessing function
def preprocess(df_train, df_test):
    df_train = df_train.loc[:, ~df_train.columns.str.startswith('Unnamed:')].copy()
    df_test = df_test.loc[:, ~df_test.columns.str.startswith('Unnamed:')].copy()
    if 'transactions' in df_train.columns:
        df_train = df_train.drop(columns=['transactions'])
    # Parse dates
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    # Time-based features
    for df in [df_train, df_test]:
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    # Label-encode categorical columns (fit on combined train+test)
    cat_cols = ['family', 'city', 'state', 'type_x', 'type_y', 'locale', 'locale_name', 'description']
    for col in cat_cols:
        if col in df_train.columns and col in df_test.columns:
            le = LabelEncoder()
            combined = pd.concat([df_train[col].astype(str), df_test[col].astype(str)], axis=0)
            le.fit(combined)
            df_train[col] = le.transform(df_train[col].astype(str))
            df_test[col] = le.transform(df_test[col].astype(str))
    # Lag features (train only)
    if 'sales' in df_train.columns:
        df_train = df_train.sort_values(['store_nbr', 'family', 'date'])
        df_train['lag_7'] = df_train.groupby(['store_nbr','family'])['sales'].shift(7)
        df_train['lag_14'] = df_train.groupby(['store_nbr','family'])['sales'].shift(14)
        df_train['lag_7'] = df_train['lag_7'].fillna(0)
        df_train['lag_14'] = df_train['lag_14'].fillna(0)
    # Scale numeric features (exclude id, target, and lags)
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if 'sales' in num_cols: num_cols.remove('sales')
    if 'id' in num_cols: num_cols.remove('id')
    if 'lag_7' in num_cols: num_cols.remove('lag_7')
    if 'lag_14' in num_cols: num_cols.remove('lag_14')
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])
    return df_train, df_test, scaler

# Button to trigger preprocessing and training
if (st.session_state['train_df'] is not None 
        and st.session_state['test_df'] is not None 
        and not st.session_state['trained']):
    if st.sidebar.button("Process & Train Models"):
        with st.spinner("Processing data and training models..."):
            df_train_raw = st.session_state['train_df']
            df_test_raw = st.session_state['test_df']
            if 'sales' not in df_train_raw.columns:
                st.error("Training data must contain a 'sales' column.")
            else:
                # Preprocess data
                try:
                    df_train_proc, df_test_proc, scaler = preprocess(df_train_raw, df_test_raw)
                except Exception as e:
                    st.error(f"Data preprocessing error: {e}")
                    df_train_proc, df_test_proc = None, None
                if df_train_proc is not None:
                    # Prepare model inputs
                    feature_cols = [c for c in df_train_proc.columns 
                                    if c not in ['id','date','sales','lag_7','lag_14']]
                    X_train = df_train_proc[feature_cols]
                    y_train = df_train_proc['sales']
                    X_test = df_test_proc[feature_cols]
                    metrics_list = []
                    preds = {}
                    # --- XGBoost ---
                    try:
                        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                        xgb_model.fit(X_train, y_train)
                        st.session_state['models']['XGBoost'] = xgb_model
                        y_pred_train = xgb_model.predict(X_train)
                        y_pred_test = xgb_model.predict(X_test)
                    except Exception:
                        y_pred_train = np.zeros(len(y_train))
                        y_pred_test = np.zeros(len(X_test))
                    preds['XGBoost'] = {'train': y_pred_train, 'test': y_pred_test}
                    # Compute metrics for XGBoost
                    y_true = y_train.values
                    y_pred_clipped = np.maximum(y_pred_train, 0)
                    try:
                        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)))
                    except Exception:
                        rmsle = None
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred_train))
                    mae = mean_absolute_error(y_true, y_pred_train)
                    metrics_list.append({'Model': 'XGBoost', 'RMSLE': rmsle, 'RMSE': rmse, 'MAE': mae})
                    # --- ARIMA on test ---
                    arima_preds = pd.Series(0.0, index=df_test_proc.index)
                    try:
                        from statsmodels.tsa.arima.model import ARIMA
                        for (store, fam), group in df_test_proc.groupby(['store_nbr','family']):
                            train_grp = df_train_proc[(df_train_proc['store_nbr']==store) & 
                                                     (df_train_proc['family']==fam)]
                            if len(train_grp) < 2:
                                continue
                            arima_model = ARIMA(train_grp['sales'], order=(1,1,1))
                            arima_res = arima_model.fit()
                            steps = len(group)
                            pred_vals = arima_res.forecast(steps=steps)
                            idx = group.sort_values('date').index
                            arima_preds.loc[idx] = pred_vals.values
                    except Exception:
                        arima_preds[:] = 0.0
                    preds['ARIMA'] = {'train': None, 'test': arima_preds.values}
                    # --- SARIMA on test ---
                    sarima_preds = pd.Series(0.0, index=df_test_proc.index)
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        for (store, fam), group in df_test_proc.groupby(['store_nbr','family']):
                            train_grp = df_train_proc[(df_train_proc['store_nbr']==store) & 
                                                     (df_train_proc['family']==fam)]
                            if len(train_grp) < 2:
                                continue
                            sarima_model = SARIMAX(train_grp['sales'], order=(1,1,1), seasonal_order=(1,0,1,7))
                            sarima_res = sarima_model.fit(disp=False)
                            steps = len(group)
                            pred_vals = sarima_res.forecast(steps=steps)
                            idx = group.sort_values('date').index
                            sarima_preds.loc[idx] = pred_vals.values
                    except Exception:
                        sarima_preds[:] = 0.0
                    preds['SARIMA'] = {'train': None, 'test': sarima_preds.values}
                    # --- Prophet on test (global) ---
                    prophet_preds = [0.0]*len(df_test_proc)
                    try:
                        from prophet import Prophet
                        prophet_df = pd.DataFrame({'ds': df_train_proc['date'], 'y': df_train_proc['sales']})
                        prophet_model = Prophet()
                        prophet_model.fit(prophet_df)
                        max_train = df_train_proc['date'].max()
                        max_test = df_test_proc['date'].max()
                        days = max((max_test - max_train).days, 0)
                        future = prophet_model.make_future_dataframe(periods=days)
                        forecast = prophet_model.predict(future)
                        pred_map = dict(zip(forecast['ds'].dt.date, forecast['yhat']))
                        prophet_preds = [float(pred_map.get(dt.date(), 0.0)) for dt in df_test_proc['date']]
                    except Exception:
                        prophet_preds = [0.0]*len(df_test_proc)
                    preds['Prophet'] = {'train': None, 'test': prophet_preds}
                    # --- LSTM on train/test ---
                    try:
                        import tensorflow as tf
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import LSTM, Dense
                        X_train_arr = X_train.values
                        y_train_arr = y_train.values
                        X_train_seq = X_train_arr.reshape((X_train_arr.shape[0], 1, X_train_arr.shape[1]))
                        lstm_model = Sequential()
                        lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train_arr.shape[1])))
                        lstm_model.add(Dense(1))
                        lstm_model.compile(optimizer='adam', loss='mse')
                        lstm_model.fit(X_train_seq, y_train_arr, epochs=5, batch_size=32, verbose=0)
                        train_pred_lstm = lstm_model.predict(X_train_seq).flatten()
                        X_test_arr = X_test.values
                        X_test_seq = X_test_arr.reshape((X_test_arr.shape[0], 1, X_test_arr.shape[1]))
                        test_pred_lstm = lstm_model.predict(X_test_seq).flatten()
                    except Exception:
                        train_pred_lstm = np.zeros(len(df_train_proc))
                        test_pred_lstm = np.zeros(len(df_test_proc))
                    preds['LSTM'] = {'train': train_pred_lstm, 'test': test_pred_lstm}
                    # Compute LSTM metrics
                    try:
                        y_pred_lstm = train_pred_lstm
                        y_pred_lstm_clipped = np.maximum(y_pred_lstm, 0)
                        rmsle_lstm = np.sqrt(mean_squared_error(np.log1p(y_true), 
                                           np.log1p(y_pred_lstm_clipped)))
                        rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
                        mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
                        metrics_list.append({'Model': 'LSTM', 'RMSLE': rmsle_lstm, 
                                              'RMSE': rmse_lstm, 'MAE': mae_lstm})
                    except Exception:
                        pass
                    # Save processed data, metrics, and predictions in session state
                    st.session_state['processed_train'] = df_train_proc
                    st.session_state['processed_test'] = df_test_proc
                    st.session_state['metrics'] = metrics_list
                    # Save train predictions for plotting
                    st.session_state['train_preds'] = {
                        'XGBoost': y_pred_train,
                        'LSTM': train_pred_lstm
                    }
                    # Prepare predictions DataFrame for test set
                    pred_df = df_test_proc[['id','date','store_nbr','family']].copy()
                    for m in preds:
                        pred_vals = preds[m]['test']
                        pred_df[m] = pred_vals
                    st.session_state['predictions'] = pred_df
                    # Mark training complete
                    st.session_state['trained'] = True
        st.success("Models trained.")
        st.experimental_rerun()

# Main Interface with tabs
if st.session_state['train_df'] is not None and st.session_state['test_df'] is not None:
    tab1, tab2 = st.tabs(["Training", "Prediction"])
    with tab1:
        st.header("Training Results")
        if st.session_state['trained']:
            # Show metrics table
            if 'metrics' in st.session_state:
                try:
                    metrics_df = pd.DataFrame(st.session_state['metrics'])
                    st.subheader("Training Metrics")
                    st.table(metrics_df)
                except Exception:
                    pass
            # Plot actual vs predicted for first 100 points (XGBoost)
            if 'processed_train' in st.session_state:
                plot_df = st.session_state['processed_train'].copy()
                plot_df['Actual'] = plot_df['sales']
                plot_df['XGBoost_Pred'] = st.session_state['train_preds']['XGBoost']
                plot_df = plot_df.sort_values('date').head(100)
                fig = px.line(plot_df, x='date', y=['Actual','XGBoost_Pred'],
                              title="Actual vs XGBoost Predicted (First 100 Rows)")
                st.plotly_chart(fig, use_container_width=True)
            # Download processed training data
            csv = st.session_state['processed_train'].to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed Training Data", csv, "processed_train.csv", "text/csv")
        else:
            st.info("Click 'Process & Train Models' in the sidebar to start training.")

    with tab2:
        st.header("Prediction (Test Set)")
        if st.session_state['trained'] and st.session_state['predictions'] is not None:
            pred_df = st.session_state['predictions']
            model_sel = st.selectbox("Select Model", 
                                     ['XGBoost','ARIMA','SARIMA','Prophet','LSTM'])
            store_sel = st.selectbox("Select Store Number", sorted(pred_df['store_nbr'].unique()))
            family_sel = st.selectbox("Select Family", sorted(pred_df['family'].unique()))
            # Filter predictions
            filtered = pred_df[(pred_df['store_nbr']==store_sel) & 
                               (pred_df['family']==family_sel)]
            if not filtered.empty:
                st.subheader(f"Predictions for Store {store_sel}, Family '{family_sel}' using {model_sel}")
                fig2 = px.line(filtered, x='date', y=model_sel, 
                               title=f"{model_sel} Predicted Sales")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(filtered[['date', model_sel]]
                             .rename(columns={'date':'Date', model_sel:'Predicted Sales'}))
            else:
                st.write("No predictions available for this selection.")
            # Download all predictions
            csv2 = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download All Predictions", csv2, "predictions.csv", "text/csv")
        else:
            st.info("Models have not been trained or no predictions available.")
