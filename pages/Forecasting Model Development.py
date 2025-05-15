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
import psutil
import warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Sales Forecasting Dashboard")

# Initialize session state
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
        
        # Data preprocessing
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
    selected_models = st.multiselect("Select Models to Train", models, default=["Naive"])
    
    # Train button
    train_button = st.button("Generate Predictions")
    
    st.divider()
    
    if train_button and train_file and test_file and sub_file and selected_models:
        with st.spinner("Processing data..."):
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
                
                # Minimal prediction generation
                for model_name in selected_models:
                    st.write(f"Generating predictions for {model_name}...")
                    temp_dir = tempfile.gettempdir()
                    val_dates = pd.date_range('2017-07-16', '2017-08-15')
                    val_steps = len(val_dates)
                    pred_dict = {}
                    actuals = []
                    preds = []
                    
                    if model_name in ["Naive", "Seasonal Naive", "Moving Average"]:
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            if model_name == "Naive":
                                pred = np.full(val_steps, group['sales'].iloc[-1])
                            elif model_name == "Seasonal Naive":
                                last_season = group['sales'].tail(7).values
                                pred = np.tile(last_season, (val_steps // 7) + 1)[:val_steps]
                            elif model_name == "Moving Average":
                                pred = np.full(val_steps, group['sales'].tail(7).mean())
                            pred_dict[(store, family)] = pred.tolist()
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(pred_dict[(store, family)])
                    
                    elif model_name == "Linear Regression":
                        X_train = train_set[feature_cols]
                        y_train = train_set['sales']
                        X_val = val_set[feature_cols]
                        y_val = val_set['sales']
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        actuals = y_val.values
                        preds = y_pred
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            mask = val_set.index.isin(group.index)
                            pred_dict[(store, family)] = y_pred[mask].tolist()
                    
                    else:
                        # Placeholder for complex models
                        for (store, family), group in train_set.groupby(['store_nbr', 'family']):
                            pred_dict[(store, family)] = np.full(val_steps, group['sales'].mean()).tolist()
                        for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                            actuals.extend(group['sales'].values)
                            preds.extend(pred_dict[(store, family)])
                    
                    # Compute metrics
                    actual = np.clip(actuals, 0, None)
                    predicted = np.clip(preds, 0, None)
                    metrics = {
                        'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                        'mae': mean_absolute_error(actual, predicted),
                        'mape': mean_absolute_percentage_error(actual + 1e-10, predicted + 1e-10)
                    }
                    
                    # Plot
                    plt.figure(figsize=(10, 5))
                    plt.plot(actuals[:100], label='Actual')
                    plt.plot(preds[:100], label='Predicted')
                    plt.title(f"{model_name} Predictions")
                    plt.legend()
                    plot_path = os.path.join(temp_dir, f"{model_name.lower()}_pred.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Store results
                    st.session_state.model_results[model_name] = {
                        'metrics': metrics,
                        'plot_path': plot_path,
                        'y_val': actuals,
                        'y_pred': pred_dict
                    }
                    
                    # Display metrics
                    st.write(f"### {model_name} Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                    col3.metric("MAE", f"{metrics['mae']:.4f}")
                    col4.metric("MAPE", f"{metrics['mape']:.4f}")
                
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
                st.metric("Memory Usage (MB)", f"{mem:.2f}")
                st.success("Predictions generated!")
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
        selected_models = st.multiselect("Select Models to Visualize", models, default=["Naive"])
        
        st.divider()
        
        if selected_models:
            for model_name in selected_models:
                st.subheader(f"{model_name} Predictions")
                if model_name in st.session_state.model_results:
                    result = st.session_state.model_results[model_name]
                    metrics = result['metrics']
                    
                    # Filter validation set
                    val_set = st.session_state.val_set
                    mask = (val_set['store_nbr'] == store_nbr) & (val_set['family'] == family)
                    if mask.sum() > 0:
                        group = val_set[mask].sort_values('date')
                        actual = group['sales'].values[:100]
                        
                        # Get predictions
                        key = (store_nbr, family)
                        pred = result['y_pred'].get(key)
                        if pred is not None and len(pred) > 0:
                            pred = np.array(pred)[:100]
                            # Plot
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
                    st.write("Model not trained. Please generate predictions in the Training tab.")
        else:
            st.write("Please select at least one model to visualize.")
    else:
        st.write("Please upload data and generate predictions in the Training tab.")
