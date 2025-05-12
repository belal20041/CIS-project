import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import os
import tempfile
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

# Streamlit page configuration
st.set_page_config(page_title="Milestone 4: MLOps, Deployment, and Monitoring", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>MILESTONE 4: MLOps, Deployment, and Monitoring</h1>", unsafe_allow_html=True)
st.markdown("This milestone implements MLOps, deploys a forecasting model, and monitors performance. Upload your data, specify columns, and generate predictions.")

# Set MLflow tracking
mlflow.set_tracking_uri("file:///D:/DEPI/CIS project/mlruns")
mlflow.set_experiment("Sales_Forecasting")

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
            df['month'] = df[date_col].dt.month.astype('int8')
            df['year'] = df[date_col].dt.year.astype('int16')
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    
    # Add lag and rolling features
    lags = [7, 14]
    for lag in lags:
        train[f'lag_{lag}'] = train.groupby(group_cols)[target_col].shift(lag).astype('float32')
        test[f'lag_{lag}'] = test.groupby(group_cols)[target_col].shift(lag).astype('float32') if target_col in test.columns else 0
    for df in [train, test]:
        roll = df.groupby(group_cols)[target_col].shift(1).rolling(7, min_periods=1)
        df['roll_mean_7'] = roll.mean().astype('float32')
    
    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        train[f'{col}_encoded'] = le.fit_transform(train[col]).astype('int8')
        test[f'{col}_encoded'] = le.transform(test[col]).astype('int8') if col in test.columns else 0
    
    # Scale numeric columns
    scaler = StandardScaler()
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols].fillna(0)).astype('float32')
    test[numeric_cols] = scaler.transform(test[numeric_cols].fillna(0)).astype('float32')
    
    return train, test, scaler

# Function to train and log XGBoost model
def train_xgboost(train, test, sub, feature_cols, target_col, id_col, date_col, train_end, val_end):
    with mlflow.start_run(run_name="XGBoost_Training"):
        # Split train and validation sets
        train_set = train[train[date_col] <= train_end]
        val_set = train[(train[date_col] > train_end) & (train[date_col] <= val_end)].dropna(subset=[target_col])
        
        if val_set.empty:
            st.error("Validation set is empty. Please check date ranges.")
            return None, None, None
        
        # Prepare features
        X_train = train_set[feature_cols]
        y_train = train_set[target_col]
        X_val = val_set[feature_cols]
        y_val = val_set[target_col]
        X_test = test[feature_cols]
        
        # Train model
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        actual = np.clip(y_val, 0, None)
        predicted = np.clip(y_pred, 0, None)
        metrics = {
            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mae': mean_absolute_error(actual, predicted),
            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
        }
        
        # Log parameters and metrics
        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "train_rows": train_set.shape[0],
            "val_rows": val_set.shape[0]
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        # Generate submission
        test_copy = test.copy()
        test_copy[target_col] = model.predict(X_test)
        submission = test_copy[[id_col, target_col]].merge(sub[[id_col]], on=id_col, how='right').fillna({target_col: 0}).clip(lower=0)
        sub_path = os.path.join("D:\\DEPI\\CIS project", "submission_xgboost.csv")
        submission.to_csv(sub_path, index=False)
        mlflow.log_artifact(sub_path)
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_val.values[:100], label='Actual')
        ax.plot(y_pred[:100], label='Predicted')
        ax.set_title("XGBoost Predictions")
        ax.legend()
        plot_path = os.path.join(tempfile.gettempdir(), "xgb_pred.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        return model, metrics, submission

# Function for real-time prediction
def real_time_prediction(model, feature_cols, input_data, scaler, categorical_encoders):
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical columns
    for col in categorical_encoders:
        if col in input_df.columns:
            input_df[f'{col}_encoded'] = categorical_encoders[col].transform(input_df[col])
    
    # Scale numeric columns
    numeric_cols = [col for col in feature_cols if col not in [f'{c}_encoded' for c in categorical_encoders]]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols].fillna(0))
    
    # Select features
    X = input_df[feature_cols]
    
    # Predict
    prediction = model.predict(X)
    return np.clip(prediction[0], 0, None)

# Function for batch prediction
def batch_prediction(model, test, sub, feature_cols, target_col, id_col, scaler, categorical_encoders):
    test_copy = test.copy()
    
    # Encode categorical columns
    for col in categorical_encoders:
        if col in test_copy.columns:
            test_copy[f'{col}_encoded'] = categorical_encoders[col].transform(test_copy[col])
    
    # Scale numeric columns
    numeric_cols = [col for col in feature_cols if col not in [f'{c}_encoded' for c in categorical_encoders]]
    test_copy[numeric_cols] = scaler.transform(test_copy[numeric_cols].fillna(0))
    
    # Predict
    test_copy[target_col] = model.predict(test_copy[feature_cols])
    submission = test_copy[[id_col, target_col]].merge(sub[[id_col]], on=id_col, how='right').fillna({target_col: 0}).clip(lower=0)
    return submission

# Function for model monitoring
def monitor_model(model, val_set, feature_cols, target_col, threshold=0.5):
    with mlflow.start_run(run_name="Model_Monitoring"):
        X_val = val_set[feature_cols]
        y_val = val_set[target_col]
        y_pred = model.predict(X_val)
        actual = np.clip(y_val, 0, None)
        predicted = np.clip(y_pred, 0, None)
        metrics = {
            'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mae': mean_absolute_error(actual, predicted),
            'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Check for drift (simple statistical test)
        drift_stat, drift_pval = ttest_ind(actual, predicted, equal_var=False)
        mlflow.log_metric("drift_stat", drift_stat)
        mlflow.log_metric("drift_pval", drift_pval)
        
        # Alert if RMSLE exceeds threshold
        alert = metrics['rmsle'] > threshold
        mlflow.log_param("alert_triggered", alert)
        
        return metrics, drift_stat, drift_pval, alert

# Main Streamlit app
def main():
    # File upload
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train Data (CSV or Parquet)", type=["csv", "parquet"])
    test_file = st.file_uploader("Upload Test Data (CSV or Parquet)", type=["csv", "parquet"])
    sub_file = st.file_uploader("Upload Submission CSV", type=["csv"])
    
    if train_file and test_file and sub_file:
        # Read files
        if train_file.name.endswith('.csv'):
            train = pd.read_csv(train_file)
        else:
            train = pd.read_parquet(train_file)
        if test_file.name.endswith('.csv'):
            test = pd.read_csv(test_file)
        else:
            test = pd.read_parquet(test_file)
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
        
        # Date range selection
        st.subheader("Specify Date Ranges")
        train_end = st.date_input("Training End Date", value=pd.to_datetime('2017-07-15'))
        val_end = st.date_input("Validation End Date", value=pd.to_datetime('2017-08-15'))
        
        if st.button("Train and Deploy Model"):
            with mlflow.start_run(run_name="Full_Pipeline"):
                # Preprocess data
                date_col = None if date_col == "None" else date_col
                train, test, scaler = preprocess_data(train, test, date_col, numeric_cols, categorical_cols, target_col, id_col, group_cols)
                
                # Define feature columns
                feature_cols = numeric_cols + [f'{col}_encoded' for col in categorical_cols] + \
                               ['day', 'dow', 'month', 'year', 'sin_month', 'lag_7', 'lag_14', 'roll_mean_7']
                feature_cols = [col for col in feature_cols if col in train.columns and col in test.columns]
                
                # Create categorical encoders
                categorical_encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    train[f'{col}_encoded'] = le.fit_transform(train[col])
                    test[f'{col}_encoded'] = le.transform(test[col]) if col in test.columns else 0
                    categorical_encoders[col] = le
                
                # Train model
                model, metrics, submission = train_xgboost(train, test, sub, feature_cols, target_col, id_col, date_col, train_end, val_end)
                
                if model:
                    # Display metrics
                    st.subheader("Model Metrics")
                    st.write(f"RMSLE: {metrics['rmsle']:.4f}")
                    st.write(f"RMSE: {metrics['rmse']:.4f}")
                    st.write(f"MAE: {metrics['mae']:.4f}")
                    st.write(f"MAPE: {metrics['mape']:.4f}")
                    
                    # Save submission
                    st.success(f"Submission saved as 'submission_xgboost.csv'")
                    
                    # Monitor model
                    val_set = train[(train[date_col] > train_end) & (train[date_col] <= val_end)]
                    monitor_metrics, drift_stat, drift_pval, alert = monitor_model(model, val_set, feature_cols, target_col)
                    st.subheader("Model Monitoring")
                    st.write(f"Monitoring RMSLE: {monitor_metrics['rmsle']:.4f}")
                    st.write(f"Drift Statistic: {drift_stat:.4f}, p-value: {drift_pval:.4f}")
                    if alert:
                        st.error("Alert: RMSLE exceeds threshold (0.5). Consider retraining the model.")
                    else:
                        st.success("Model performance is within acceptable limits.")
                
                # Real-time prediction interface
                st.subheader("Real-Time Prediction")
                input_data = {}
                for col in feature_cols:
                    if col.endswith('_encoded'):
                        orig_col = col.replace('_encoded', '')
                        input_data[orig_col] = st.selectbox(f"Select {orig_col}", train[orig_col].unique())
                    else:
                        input_data[col] = st.number_input(f"Enter {col}", value=0.0)
                
                if st.button("Predict"):
                    prediction = real_time_prediction(model, feature_cols, input_data, scaler, categorical_encoders)
                    st.write(f"Predicted {target_col}: {prediction:.2f}")
                
                # Batch prediction interface
                st.subheader("Batch Prediction")
                batch_file = st.file_uploader("Upload Batch Test Data (CSV)", type=["csv"], key="batch")
                if batch_file:
                    batch_test = pd.read_csv(batch_file)
                    batch_submission = batch_prediction(model, batch_test, sub, feature_cols, target_col, id_col, scaler, categorical_encoders)
                    batch_sub_path = os.path.join("D:\\DEPI\\CIS project", "submission_batch_xgboost.csv")
                    batch_submission.to_csv(batch_sub_path, index=False)
                    st.success(f"Batch submission saved as 'submission_batch_xgboost.csv'")
                    mlflow.log_artifact(batch_sub_path)

if __name__ == "__main__":
    main()
