import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Sales Forecasting Dashboard")
st.write("Upload your sales data and train or apply forecasting models interactively.")

# --- Sidebar: File Upload ---
st.sidebar.header("Upload CSV Files")
train_file = st.sidebar.file_uploader("Training CSV", type=['csv'], key='train_csv')
test_file = st.sidebar.file_uploader("Test CSV", type=['csv'], key='test_csv')

# Initialize session state for data storage
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

# Load data into session state on upload
if train_file is not None:
    st.session_state.train_df = pd.read_csv(train_file)
if test_file is not None:
    st.session_state.test_df = pd.read_csv(test_file)

# Helper to clean and validate a DataFrame
def clean_and_validate(df, name):
    # Drop any 'Unnamed:' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Required columns
    required = ['id','date','store_nbr','family','sales','onpromotion']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{name}: missing required columns: {missing}")
        return None
    # Ensure date column is datetime
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            st.error(f"{name}: could not parse 'date' column as datetime.")
            return None
    return df

# Clean/validate uploaded data
train_df = None
test_df = None
if st.session_state.train_df is not None:
    train_df = clean_and_validate(st.session_state.train_df.copy(), "Training data")
if st.session_state.test_df is not None:
    test_df = clean_and_validate(st.session_state.test_df.copy(), "Test data")

# Proceed only if training data is loaded and valid
if train_df is not None:
    # Tabs for Training and Prediction
    tab1, tab2 = st.tabs(["Training", "Prediction"])

    # ----------- Training Tab -----------
    with tab1:
        st.header("Training")
        # Show first few rows of training data
        st.subheader("Training Data Preview")
        st.dataframe(train_df.head(5))

        # Allow user to select date and target columns
        date_col = st.selectbox("Select Date Column", options=train_df.columns, index=list(train_df.columns).index('date') if 'date' in train_df.columns else 0)
        target_col = st.selectbox("Select Target (Sales) Column", options=[col for col in train_df.columns if train_df[col].dtype in [np.int64, np.float64]], index=list(train_df.columns).index('sales') if 'sales' in train_df.columns else 0)

        # When ready, start preprocessing and training
        if st.button("Preprocess & Train Models"):
            with st.spinner("Preprocessing data..."):
                df = train_df.copy()
                # Rename selected columns to standard names
                df.rename(columns={date_col: 'date', target_col: 'sales'}, inplace=True)
                # Sort by date for time series order
                df.sort_values('date', inplace=True)
                # Label encode categorical columns
                cat_cols = df.select_dtypes(include=['object']).columns.drop(['id','date'])
                for col in cat_cols:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                # Feature engineering: date parts
                df['month']   = df['date'].dt.month
                df['day']     = df['date'].dt.day
                df['weekday'] = df['date'].dt.weekday
                df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
                # Lag features (7-day and 14-day)
                df['lag_7']  = df.groupby(['store_nbr','family'])['sales'].shift(7)
                df['lag_14'] = df.groupby(['store_nbr','family'])['sales'].shift(14)
                df.dropna(subset=['lag_7','lag_14'], inplace=True)
                # Scaling numeric features (except target)
                num_cols = df.select_dtypes(include=[np.number]).columns.drop(['id','sales'])
                scaler = StandardScaler()
                df[num_cols] = scaler.fit_transform(df[num_cols])
                st.success("Preprocessing complete.")
                # Save processed training data
                st.session_state.processed_train = df

            # Model Training
            with st.spinner("Training models..."):
                X = df.drop(columns=['id','date','sales'])
                y = df['sales'].values

                # --- XGBoost Model ---
                xgb_model = XGBRegressor(tree_method='hist', verbosity=0)
                xgb_model.fit(X, y)
                df['pred_xgb'] = xgb_model.predict(X)

                # --- Prophet Model ---
                prophet_df = df[['date','sales']].rename(columns={'date':'ds','sales':'y'})
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=0)  # no extra periods since forecasting training range
                forecast = m.predict(future)
                df['pred_prophet'] = forecast['yhat'].values

                # --- ARIMA/SARIMA Model ---
                # For simplicity, fit on the aggregated series of all stores as an example
                try:
                    arima_model = SARIMAX(df['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                    arima_res = arima_model.fit(disp=False)
                    df['pred_arima'] = arima_res.predict(start=0, end=len(df)-1)
                except Exception as e:
                    st.error(f"ARIMA training failed: {e}")
                    df['pred_arima'] = np.nan

                # --- LSTM Model ---
                # Reshape X for LSTM: (samples, timesteps=1, features)
                X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
                lstm_model = Sequential([LSTM(32, return_sequences=False, input_shape=(1, X.shape[1])), Dense(1)])
                lstm_model.compile(optimizer='adam', loss='mse')
                lstm_model.fit(X_lstm, y, epochs=10, verbose=0)
                df['pred_lstm'] = lstm_model.predict(X_lstm).flatten()

                st.success("Models trained.")

                # Calculate metrics
                def calc_metrics(y_true, y_pred):
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae  = mean_absolute_error(y_true, y_pred)
                    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))
                    return rmse, mae, rmsle

                metrics = {}
                for model in ['xgb','prophet','arima','lstm']:
                    if f'pred_{model}' in df:
                        rmse, mae, rmsle = calc_metrics(df['sales'], df[f'pred_{model}'])
                        metrics[model] = {'RMSE': rmse, 'MAE': mae, 'RMSLE': rmsle}

                st.subheader("Training Metrics")
                # Display metrics table
                if metrics:
                    met_df = pd.DataFrame(metrics).T[['RMSE','MAE','RMSLE']]
                    st.table(met_df)

                # Plot actual vs predicted (first 100 points)
                st.subheader("Actual vs Predicted (first 100 points)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'].head(100), y=df['sales'].head(100),
                    mode='lines', name='Actual'))
                # Plot predictions for each model
                colors = {'pred_xgb':'blue','pred_prophet':'green','pred_arima':'orange','pred_lstm':'purple'}
                for col in ['pred_xgb','pred_prophet','pred_arima','pred_lstm']:
                    if col in df:
                        fig.add_trace(go.Scatter(
                            x=df['date'].head(100), y=df[col].head(100),
                            mode='lines', name=col.split('_')[1].title(),
                            line=dict(color=colors.get(col, None), dash='dot')))
                st.plotly_chart(fig, use_container_width=True)

                # Download buttons for processed data and predictions
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed Training Data",
                    data=csv_data, file_name="processed_training_data.csv", mime="text/csv"
                )
                # Prepare prediction output (id + preds)
                pred_cols = ['id'] + [f'pred_{m}' for m in ['xgb','prophet','arima','lstm'] if f'pred_{m}' in df]
                preds_df = df[pred_cols].copy()
                preds_csv = preds_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Training Predictions",
                    data=preds_csv, file_name="training_predictions.csv", mime="text/csv"
                )

    # ----------- Prediction Tab -----------
    with tab2:
        st.header("Prediction")
        if test_df is None:
            st.info("Upload and validate a Test CSV to enable predictions.")
        else:
            st.subheader("Test Data Preview")
            st.dataframe(test_df.head(5))
            # Filters
            store_vals = sorted(test_df['store_nbr'].unique())
            fam_vals   = sorted(test_df['family'].unique())
            model_choice = st.selectbox("Select Model", ["XGBoost","Prophet","ARIMA","LSTM"])
            store_choice = st.selectbox("Select Store Number", store_vals)
            fam_choice   = st.selectbox("Select Product Family", fam_vals)
            if st.button("Generate Forecast"):
                subset = test_df[(test_df['store_nbr']==store_choice) & (test_df['family']==fam_choice)].copy()
                if subset.empty:
                    st.warning("No data for selected store/family.")
                else:
                    # Apply same preprocessing as train
                    subset['date'] = pd.to_datetime(subset['date'])
                    subset.sort_values('date', inplace=True)
                    for col in cat_cols:  # reuse training cat_cols
                        if col in subset:
                            subset[col] = LabelEncoder().fit_transform(subset[col].astype(str))
                    subset['month']   = subset['date'].dt.month
                    subset['day']     = subset['date'].dt.day
                    subset['weekday'] = subset['date'].dt.weekday
                    subset['is_weekend'] = subset['weekday'].isin([5,6]).astype(int)
                    subset['lag_7']  = subset.groupby(['store_nbr','family'])['sales'].shift(7)
                    subset['lag_14'] = subset.groupby(['store_nbr','family'])['sales'].shift(14)
                    subset.dropna(inplace=True)
                    subset[num_cols] = scaler.transform(subset[num_cols])
                    # Predict with chosen model
                    if model_choice == "XGBoost":
                        y_pred = xgb_model.predict(subset.drop(columns=['id','date','sales']))
                    elif model_choice == "Prophet":
                        temp = subset[['date','sales']].rename(columns={'date':'ds','sales':'y'})
                        future = m.make_future_dataframe(periods=0)
                        forecast = m.predict(future)
                        y_pred = forecast['yhat'].values[-len(subset):]
                    elif model_choice == "ARIMA":
                        try:
                            res = arima_res
                            y_pred = res.predict(start=0, end=len(subset)-1)
                        except:
                            y_pred = np.full(len(subset), np.nan)
                    elif model_choice == "LSTM":
                        X_pred = subset.drop(columns=['id','date','sales']).values.reshape((len(subset),1,len(num_cols)))
                        y_pred = lstm_model.predict(X_pred).flatten()
                    else:
                        y_pred = np.full(len(subset), np.nan)
                    subset['pred'] = y_pred
                    # Plot
                    st.subheader(f"{model_choice} Forecast for Store {store_choice}, Family '{fam_choice}'")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=subset['date'], y=subset['sales'], mode='lines', name='Actual'))
                    fig2.add_trace(go.Scatter(x=subset['date'], y=subset['pred'], mode='lines', name='Predicted'))
                    st.plotly_chart(fig2, use_container_width=True)
                    # Group metrics
                    rmse, mae, rmsle = calc_metrics(subset['sales'], subset['pred'])
                    st.write(f"**Metrics (subset)** – RMSE: {rmse:.3f}, MAE: {mae:.3f}, RMSLE: {rmsle:.3f}")

