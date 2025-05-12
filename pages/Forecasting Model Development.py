import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Streamlit page configuration
st.set_page_config(page_title="Milestone 3: Model Training & Forecasting", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>MILESTONE 3: Model Training & Forecasting</h1>", unsafe_allow_html=True)
st.markdown("This milestone trains forecasting models and generates predictions. Upload your data, specify columns, and select models to proceed.")

# Function to preprocess data
def preprocess_data(train, test, date_col, numeric_cols, categorical_cols, target_col, id_col, group_cols):
    train = train.copy()
    test = test.copy()
    
    # Standardize column names
    for df in [train, test]:
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    
    # Convert date column to datetime
    if date_col:
        train[date_col] = pd.to_datetime(train[date_col], errors='coerce')
        test[date_col] = pd.to_datetime(test[date_col], errors='coerce')
    
    # Convert categorical columns to category dtype
    for col in categorical_cols:
        if col in train.columns:
            train[col] = train[col].astype('category')
        if col in test.columns:
            test[col] = test[col].astype('category')
    
    # Ensure numeric columns are numeric
    for col in numeric_cols:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors='coerce')
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors='coerce')
    
    # Handle missing values
    train = train.fillna(0)
    test = test.fillna(0)
    
    # Add time-based features
    if date_col:
        for df in [train, test]:
            df['day'] = df[date_col].dt.day.astype('int8')
            df['dow'] = df[date_col].dt.dayofweek.astype('int8')
            df['is_weekend'] = df['dow'].isin([5, 6]).astype('int8')
            df['woy'] = df[date_col].dt.isocalendar().week.astype('int8')
            df['month'] = df[date_col].dt.month.astype('int8')
            df['quarter'] = df[date_col].dt.quarter.astype('int8')
            df['year'] = df[date_col].dt.year.astype('int16')
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12).astype('float32')
    
    # Add lag and rolling features
    lags = [1, 7, 14, 21, 28]
    windows = [7, 14, 21, 28]
    for lag in lags:
        train[f'lag_{lag}'] = train.groupby(group_cols)[target_col].shift(lag).astype('float32')
        test[f'lag_{lag}'] = test.groupby(group_cols)[target_col].shift(lag).astype('float32') if target_col in test.columns else 0
    for w in windows:
        for df in [train, test]:
            roll = df.groupby(group_cols)[target_col].shift(1).rolling(w, min_periods=1)
            df[f'roll_mean_{w}'] = roll.mean().astype('float32')
            df[f'roll_std_{w}'] = roll.std().astype('float32')
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        train[f'{col}_encoded'] = le.fit_transform(train[col]).astype('int8')
        test[f'{col}_encoded'] = le.transform(test[col]).astype('int8') if col in test.columns else 0
    
    # Scale numeric columns
    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols].fillna(0)).astype('float32')
    test[numeric_cols] = scaler.transform(test[numeric_cols].fillna(0)).astype('float32')
    
    return train, test

# Function to train and evaluate models
def train_models(train, test, sub, feature_cols, target_col, id_col, group_cols, date_col, train_end, val_end, models):
    metrics = {}
    submissions = {}
    
    # Split train and validation sets
    train_set = train[train[date_col] <= train_end]
    val_set = train[(train[date_col] > train_end) & (train[date_col] <= val_end)].dropna(subset=[target_col])
    
    if val_set.empty:
        st.error("Validation set is empty. Please check date ranges.")
        return None, None
    
    # Prepare features
    X_train = train_set[feature_cols]
    y_train = train_set[target_col]
    X_val = val_set[feature_cols]
    y_val = val_set[target_col]
    X_test = test[feature_cols]
    
    # XGBoost
    if 'xgboost' in models:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='rmse'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred_val = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        actual = np.clip(y_val, 0, None)
        predicted = np.clip(y_pred_val, 0, None)
        metrics['xgboost'] = {
            'RMSLE': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
        }
        
        test_copy = test.reset_index()
        test_copy[target_col] = test_preds
        submission = test_copy[[id_col, target_col]].merge(sub[[id_col]], on=id_col, how='right').fillna({target_col: 0}).clip(lower=0)
        submissions['xgboost'] = submission
    
    # ARIMA
    if 'arima' in models:
        train_groups = train_set.groupby(group_cols)
        val_groups = val_set.groupby(group_cols)
        val_dates = pd.date_range(val_end, periods=(pd.to_datetime(val_end) - pd.to_datetime(train_end)).days + 1)
        test_dates = pd.date_range(val_end + pd.Timedelta(days=1), periods=16)  # Assuming 16-day test period
        val_steps = len(val_dates)
        test_steps = len(test_dates)
        
        arima_models = {}
        for key, group in train_groups:
            try:
                model = auto_arima(group[target_col], seasonal=False, max_p=5, max_q=5, trace=False, error_action='ignore')
                arima_models[key] = model
            except:
                arima_models[key] = None
        
        arima_val_preds = {}
        for k, m in arima_models.items():
            arima_val_preds[k] = m.predict(val_steps) if m else np.zeros(val_steps)
        
        actuals = []
        preds = []
        for (s, f), g in val_groups:
            actuals.extend(g[target_col].values)
            preds.extend(arima_val_preds[(s, f)])
        
        actual = np.clip(actuals, 0, None)
        predicted = np.clip(preds, 0, None)
        metrics['arima'] = {
            'RMSLE': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
        }
        
        arima_test_preds = {}
        for k, m in arima_models.items():
            arima_test_preds[k] = m.predict(test_steps) if m else np.zeros(test_steps)
        
        test_copy = test.reset_index()
        for (store, family), pred_values in arima_test_preds.items():
            mask = (test_copy[group_cols[0]] == store) & (test_copy[group_cols[1]] == family)
            test_copy.loc[mask, target_col] = pred_values[:mask.sum()]
        submission = test_copy[[id_col, target_col]].merge(sub[[id_col]], on=id_col, how='right').fillna({target_col: 0}).clip(lower=0)
        submissions['arima'] = submission
    
    # Prophet
    if 'prophet' in models:
        train_groups = train_set.groupby(group_cols)
        val_groups = val_set.groupby(group_cols)
        val_dates = pd.date_range(val_end, periods=(pd.to_datetime(val_end) - pd.to_datetime(train_end)).days + 1)
        test_dates = pd.date_range(val_end + pd.Timedelta(days=1), periods=16)
        
        prophet_models = {}
        for key, group in train_groups:
            df = group.reset_index()[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            m.fit(df)
            prophet_models[key] = m
        
        prophet_val_preds = {}
        for k, m in prophet_models.items():
            prophet_val_preds[k] = m.predict(pd.DataFrame({'ds': val_dates}))['yhat'].values
        
        actuals = []
        preds = []
        for (s, f), g in val_groups:
            actuals.extend(g[target_col].values)
            preds.extend(prophet_val_preds[(s, f)])
        
        actual = np.clip(actuals, 0, None)
        predicted = np.clip(preds, 0, None)
        metrics['prophet'] = {
            'RMSLE': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
        }
        
        prophet_test_preds = {}
        for k, m in prophet_models.items():
            prophet_test_preds[k] = m.predict(pd.DataFrame({'ds': test_dates}))['yhat'].values
        
        test_copy = test.reset_index()
        for (store, family), pred_values in prophet_test_preds.items():
            mask = (test_copy[group_cols[0]] == store) & (test_copy[group_cols[1]] == family)
            test_copy.loc[mask, target_col] = pred_values[:mask.sum()]
        submission = test_copy[[id_col, target_col]].merge(sub[[id_col]], on=id_col, how='right').fillna({target_col: 0}).clip(lower=0)
        submissions['prophet'] = submission
    
    return metrics, submissions

# Function to plot time-series analysis
def plot_time_series_analysis(df, target_col, group_cols, group_values):
    ts = df.loc[tuple(group_values), target_col].sort_index()
    
    st.subheader(f"Time Series Analysis for {group_cols[0]}={group_values[0]}, {group_cols[1]}={group_values[1]}")
    
    # Plot original series, rolling mean, and std
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, label='Original')
    ax.plot(ts.rolling(30).mean(), label='Rolling Mean', color='red')
    ax.plot(ts.rolling(30).std(), label='Rolling Std', color='black')
    ax.set_title(f"Stationarity: {group_cols[0]}={group_values[0]}, {group_cols[1]}={group_values[1]}")
    ax.legend()
    st.pyplot(fig)
    
    # ADF test
    result = adfuller(ts.dropna())
    st.write(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    
    # ACF plot
    ts_diff = ts.diff().dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(ts_diff, lags=20, ax=ax)
    ax.set_title("ACF of Differenced Series")
    st.pyplot(fig)

# Main Streamlit app
def main():
    # File upload
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type=["csv"])
    test_file = st.file_uploader("Upload Test CSV", type=["csv"])
    sub_file = st.file_uploader("Upload Submission CSV", type=["csv"])
    
    if train_file and test_file and sub_file:
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        sub = pd.read_csv(sub_file)
        
        # Column selection
        st.subheader("Specify Columns")
        all_cols = train.columns.tolist()
        date_col = st.selectbox("Select Date Column", ["None"] + all_cols, index=0)
        target_col = st.selectbox("Select Target Column", all_cols)
        id_col = st.selectbox("Select ID Column", all_cols)
        group_cols = st.multiselect("Select Grouping Columns (e.g., store_nbr, family)", all_cols, default=['store_nbr', 'family'])
        numeric_cols = st.multiselect("Select Numeric Columns", all_cols)
        categorical_cols = st.multiselect("Select Categorical Columns", all_cols)
        
        # Model selection
        st.subheader("Select Models")
        models = st.multiselect("Select Models to Train", ['xgboost', 'arima', 'prophet'], default=['xgboost'])
        
        # Date range selection
        st.subheader("Specify Date Ranges")
        train_end = st.date_input("Training End Date", value=pd.to_datetime('2017-07-15'))
        val_end = st.date_input("Validation End Date", value=pd.to_datetime('2017-08-15'))
        
        if st.button("Train Models"):
            # Preprocess data
            date_col = None if date_col == "None" else date_col
            train, test = preprocess_data(train, test, date_col, numeric_cols, categorical_cols, target_col, id_col, group_cols)
            
            # Define feature columns
            feature_cols = numeric_cols + [f'{col}_encoded' for col in categorical_cols] + \
                           ['day', 'dow', 'is_weekend', 'woy', 'month', 'quarter', 'year', 'sin_month', 'cos_month'] + \
                           [f'lag_{lag}' for lag in [1, 7, 14, 21, 28]] + \
                           [f'roll_mean_{w}' for w in [7, 14, 21, 28]] + [f'roll_std_{w}' for w in [7, 14, 21, 28]]
            feature_cols = [col for col in feature_cols if col in train.columns and col in test.columns]
            
            # Train models
            metrics, submissions = train_models(train, test, sub, feature_cols, target_col, id_col, group_cols, date_col, train_end, val_end, models)
            
            if metrics and submissions:
                # Display metrics
                st.subheader("Model Metrics")
                for model_name, model_metrics in metrics.items():
                    st.write(f"{model_name.upper()} Metrics:")
                    st.write(f"RMSLE: {model_metrics['RMSLE']:.4f}")
                    st.write(f"RMSE: {model_metrics['RMSE']:.4f}")
                    st.write(f"MAE: {model_metrics['MAE']:.4f}")
                    st.write(f"MAPE: {model_metrics['MAPE']:.4f}")
                
                # Save submissions
                for model_name, submission in submissions.items():
                    submission.to_csv(f"D:\\DEPI\\CIS project\\submission_{model_name}.csv", index=False)
                    st.success(f"Submission saved as 'submission_{model_name}.csv'")
                
                # Save processed data
                train_set = train[train[date_col] <= train_end]
                val_set = train[(train[date_col] > train_end) & (train[date_col] <= val_end)]
                train_set.select_dtypes(include=[np.number]).to_parquet("D:\\DEPI\\CIS project\\train_processed.parquet")
                val_set.select_dtypes(include=[np.number]).to_parquet("D:\\DEPI\\CIS project\\val_processed.parquet")
                test.select_dtypes(include=[np.number]).to_parquet("D:\\DEPI\\CIS project\\test_processed.parquet")
                st.success("Processed data saved as Parquet files.")
                
                # Time-series analysis
                st.subheader("Time Series Analysis")
                group_values = st.selectbox("Select Group for Analysis", train[group_cols].drop_duplicates().values.tolist())
                train.set_index(group_cols + [date_col], inplace=True)
                plot_time_series_analysis(train, target_col, group_cols, group_values)

if __name__ == "__main__":
    main()
