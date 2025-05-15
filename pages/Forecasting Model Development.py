import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import joblib
import psutil
from datetime import datetime

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
training_tab, prediction_tab, specific_prediction_tab, forecasting_tab = st.tabs(["Training", "Prediction", "Specific Date Prediction", "Forecasting"])

# Constants
TRAIN_END = '2017-07-15'
VAL_END = '2017-08-15'

# Custom MAPE to avoid division by zero
def clipped_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Cache data loading and processing
@st.cache_data(hash_funcs={st.file_uploader: lambda x: x.name})
def load_and_process_data(train_file, test_file, sub_file):
    # Load data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    sub = pd.read_csv(sub_file)
    
    # Parse dates with flexible format
    train['date'] = pd.to_datetime(train['date'], errors='coerce')
    test['date'] = pd.to_datetime(test['date'], errors='coerce')
    train = train.dropna(subset=['date'])
    test = test.dropna(subset=['date'])
    
    # Type conversions
    train[['store_nbr', 'onpromotion']] = train[['store_nbr', 'onpromotion']].astype('int32')
    test[['store_nbr', 'onpromotion']] = test[['store_nbr', 'onpromotion']].astype('int32')
    train['sales'] = train['sales'].astype('float32')
    
    # Combine and aggregate
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', 'date'])
    agg_dict = {'sales': 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
    combined = combined.groupby(['store_nbr', 'family', 'date']).agg(agg_dict).reset_index()
    combined = combined.astype({'store_nbr': 'int32', 'family': 'category', 'date': 'datetime64[ns]', 
                               'sales': 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
    
    # Handle missing sales with interpolation
    grouped = combined.groupby(['store_nbr', 'family'])
    processed_groups = []
    for (store_nbr, family), group in grouped:
        group['sales'] = group['sales'].interpolate(method='linear', limit_direction='both').fillna(0).astype('float32')
        group['onpromotion'] = group['onpromotion'].fillna(0).astype('int32')
        processed_groups.append(group)
    combined = pd.concat(processed_groups).sort_values(['store_nbr', 'family', 'date'])
    
    # Add features
    combined['day'] = combined['date'].dt.day.astype('int8')
    combined['dow'] = combined['date'].dt.dayofweek.astype('int8')
    combined['month'] = combined['date'].dt.month.astype('int8')
    combined['year'] = combined['date'].dt.year.astype('int16')
    combined['sin_month'] = np.sin(2 * np.pi * combined['month'] / 12).astype('float32')
    combined['cos_month'] = np.cos(2 * np.pi * combined['month'] / 12).astype('float32')
    lags = [7, 14, 28]
    for lag in lags:
        combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag).fillna(0).astype('float32')
    combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).mean().fillna(0).astype('float32')
    combined['roll_std_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).std().fillna(0).astype('float32')
    
    # Encode categorical variables
    le_store = LabelEncoder()
    le_family = LabelEncoder()
    combined['store_nbr_encoded'] = le_store.fit_transform(combined['store_nbr']).astype('int8')
    combined['family_encoded'] = le_family.fit_transform(combined['family']).astype('int8')
    
    # Define features
    feature_cols = ['onpromotion', 'day', 'dow', 'month', 'year', 'sin_month', 'cos_month', 
                    'store_nbr_encoded', 'family_encoded', 'lag_7', 'lag_14', 'lag_28', 
                    'roll_mean_7', 'roll_std_7']
    
    # Scale features
    scaler = StandardScaler()
    combined[feature_cols] = scaler.fit_transform(combined[feature_cols]).astype('float32')
    
    # Split data
    train = combined[combined['is_train'] == 1]
    test = combined[combined['is_train'] == 0].drop(['sales'], axis=1)
    train_set = train[train['date'] <= TRAIN_END]
    val_set = train[(train['date'] > TRAIN_END) & (train['date'] <= VAL_END)]
    
    return train_set, val_set, test, sub, feature_cols, scaler, le_store, le_family

# Training Tab
with training_tab:
    st.header("Train Forecasting Models")
    
    # File uploaders
    st.subheader("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")
    sub_file = st.file_uploader("Upload Submission CSV", type="csv", key="uploader_sub")
    
    # Model selection
    models = ["Naive", "Seasonal Naive", "Moving Average", "Ridge Regression"]
    selected_models = st.multiselect("Select Models to Train", models, default=["Naive"])
    
    # Train button
    train_button = st.button("Generate Predictions")
    
    st.divider()
    
    if train_button and train_file and test_file and sub_file and selected_models:
        with st.spinner("Processing data..."):
            # Load and process data
            train_set, val_set, test, sub, feature_cols, scaler, le_store, le_family = load_and_process_data(train_file, test_file, sub_file)
            st.session_state.train_set = train_set
            st.session_state.val_set = val_set
            st.session_state.test = test
            st.session_state.sub = sub
            st.session_state.feature_cols = feature_cols
            st.session_state.scaler = scaler
            st.session_state.le_store = le_store
            st.session_state.le_family = le_family
            
            # Prediction generation
            for model_name in selected_models:
                st.write(f"Generating predictions for {model_name}...")
                temp_dir = tempfile.gettempdir()
                val_dates = pd.date_range('2017-07-16', '2017-08-15')
                pred_dict = {}
                actuals = []
                preds = []
                
                # Initialize model_results
                st.session_state.model_results[model_name] = {
                    'metrics': None,
                    'plot_path': None,
                    'y_val': None,
                    'y_pred': None
                }
                
                if model_name in ["Naive", "Seasonal Naive", "Moving Average"]:
                    for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                        train_group = train_set[(train_set['store_nbr'] == store) & (train_set['family'] == family)]
                        group_sales = group['sales'].values
                        if not train_group.empty:
                            if model_name == "Naive":
                                last_sale = train_group['sales'].iloc[-1]
                                pred = np.full(len(group_sales), last_sale)
                            elif model_name == "Seasonal Naive":
                                last_week = train_group['sales'].tail(7).values if len(train_group) >= 7 else np.full(7, train_group['sales'].mean())
                                pred = np.tile(last_week, (len(group_sales) // 7) + 1)[:len(group_sales)]
                            elif model_name == "Moving Average":
                                window = min(7, len(train_group))
                                ma_value = train_group['sales'].tail(window).mean()
                                pred = np.full(len(group_sales), ma_value)
                        else:
                            pred = np.zeros(len(group_sales))
                        pred_dict[(store, family)] = pred.tolist()
                        actuals.extend(group_sales)
                        preds.extend(pred)
                
                elif model_name == "Ridge Regression":
                    X_train = train_set[feature_cols]
                    y_train = np.log1p(train_set['sales'].clip(0))
                    X_val = val_set[feature_cols]
                    y_val = val_set['sales']
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    y_pred_log = model.predict(X_val)
                    y_pred = np.expm1(y_pred_log).clip(0)
                    actuals = y_val.values
                    preds = y_pred
                    for (store, family), group in val_set.groupby(['store_nbr', 'family']):
                        mask = val_set.index.isin(group.index)
                        pred_dict[(store, family)] = y_pred[mask].tolist()
                    
                    # Save model weights as model.pt
                    model_path = os.path.join(temp_dir, "model.pt")
                    joblib.dump(model, model_path)
                    st.session_state.model_results[model_name]['model_path'] = model_path
                
                # Update model_results
                st.session_state.model_results[model_name]['y_val'] = actuals
                st.session_state.model_results[model_name]['y_pred'] = pred_dict
                
                # Compute metrics
                actual = np.clip(actuals, 0, None)
                predicted = np.clip(preds, 0, None)
                metrics = {
                    'rmsle': np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(predicted))),
                    'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                    'mae': mean_absolute_error(actual, predicted),
                    'mape': clipped_mape(actual, predicted)
                }
                
                # Plot
                plt.figure(figsize=(10, 5))
                plt.plot(val_dates[:len(actuals[:100])], actuals[:100], label='Actual')
                plt.plot(val_dates[:len(preds[:100])], preds[:100], label='Predicted')
                plt.title(f"{model_name} Predictions")
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(temp_dir, f"{model_name.lower()}_pred.png")
                plt.savefig(plot_path)
                plt.close()
                
                # Update results
                st.session_state.model_results[model_name]['metrics'] = metrics
                st.session_state.model_results[model_name]['plot_path'] = plot_path
                
                # Display metrics
                st.write(f"### {model_name} Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                col3.metric("MAE", f"{metrics['mae']:.4f}")
                col4.metric("MAPE (%)", f"{metrics['mape']:.2f}")
            
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
            st.metric("Memory Usage (MB)", f"{mem:.2f}")
            st.success("Predictions generated!")
    elif train_button:
        st.error("Please upload all CSV files and select at least one model.")

# Prediction Tab
with prediction_tab:
    st.header("Visualize Predictions")
    
    if st.session_state.train_set is not None:
        # Get store numbers and families
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        # User inputs for visualization
        st.subheader("Visualize Predictions vs Actual")
        store_nbr = st.selectbox("Select Store Number", store_nbrs, key="viz_store")
        family = st.selectbox("Select Product Family", families, key="viz_family")
        selected_models = st.multiselect("Select Models to Visualize", models, default=["Naive"], key="viz_models")
        
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
                        actual = group['sales'].values
                        dates = group['date'].values
                        
                        # Get predictions
                        key = (store_nbr, family)
                        pred = result['y_pred'].get(key)
                        if pred is not None and len(pred) > 0:
                            pred = np.array(pred)
                            # Plot
                            plt.figure(figsize=(10, 5))
                            plt.plot(dates[:100], actual[:100], label='Actual', color='blue')
                            plt.plot(dates[:100], pred[:100], label='Predicted', color='orange')
                            plt.title(f"{model_name} Predictions: Store {store_nbr}, Family {family}")
                            plt.xlabel("Date")
                            plt.ylabel("Sales")
                            plt.legend()
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plot_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_custom_pred.png")
                            plt.savefig(plot_path)
                            plt.close()
                            
                            # Display plot
                            image = Image.open(plot_path)
                            st.image(image, caption=f"{model_name} Predictions vs Actual", use_column_width=True)
                            
                            # Display metrics
                            st.write("### Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                            col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                            col3.metric("MAE", f"{metrics['mae']:.4f}")
                            col4.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                        else:
                            st.write("Predictions not available for this store-family combination.")
                    else:
                        st.write("No validation data for this store-family combination.")
                else:
                    st.write("Model not trained. Please generate predictions in the Training tab.")
    else:
        st.write("Please upload data and generate predictions in the Training tab.")

# Specific Date Prediction Tab
with specific_prediction_tab:
    st.header("Predict Sales for Specific Date")
    
    if st.session_state.train_set is not None:
        # Get store numbers and families
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        # User inputs
        st.subheader("Specify Prediction Details")
        store_nbr = st.selectbox("Select Store Number", store_nbrs, key="spec_store")
        family = st.selectbox("Select Product Family", families, key="spec_family")
        onpromotion = st.selectbox("On Promotion?", [0, 1], key="spec_promo")
        time_granularity = st.selectbox("Select Time Granularity", ["Day", "Month", "Year"], key="spec_time")
        
        if time_granularity == "Day":
            target_date = st.date_input("Select Date", min_value=datetime(2017, 8, 16), key="spec_date")
            target_dates = [pd.to_datetime(target_date)]
        elif time_granularity == "Month":
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="spec_year")
            month = st.number_input("Select Month", min_value=1, max_value=12, value=8, key="spec_month")
            target_dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-28", freq='D')
        else:  # Year
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="spec_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        predict_button = st.button("Predict Sales")
        
        if predict_button:
            with st.spinner("Generating prediction..."):
                # Prepare data for prediction
                spec_data = []
                for date in target_dates:
                    spec_data.append({
                        'store_nbr': store_nbr,
                        'family': family,
                        'date': date,
                        'onpromotion': onpromotion,
                        'is_train': 0
                    })
                spec_df = pd.DataFrame(spec_data)
                
                # Add features
                spec_df['day'] = spec_df['date'].dt.day.astype('int8')
                spec_df['dow'] = spec_df['date'].dt.dayofweek.astype('int8')
                spec_df['month'] = spec_df['date'].dt.month.astype('int8')
                spec_df['year'] = spec_df['date'].dt.year.astype('int16')
                spec_df['sin_month'] = np.sin(2 * np.pi * spec_df['month'] / 12).astype('float32')
                spec_df['cos_month'] = np.cos(2 * np.pi * spec_df['month'] / 12).astype('float32')
                spec_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(spec_df)).astype('int8')
                spec_df['family_encoded'] = st.session_state.le_family.transform([family] * len(spec_df)).astype('int8')
                
                # Compute lags and rolling means
                train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                        (st.session_state.train_set['family'] == family)]
                combined = pd.concat([train_group, spec_df]).sort_values(['store_nbr', 'family', 'date'])
                for lag in [7, 14, 28]:
                    combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag).fillna(0).astype('float32')
                combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).mean().fillna(0).astype('float32')
                combined['roll_std_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).std().fillna(0).astype('float32')
                spec_df = combined[combined['date'].isin(target_dates)]
                
                # Scale features
                spec_df[st.session_state.feature_cols] = st.session_state.scaler.transform(spec_df[st.session_state.feature_cols]).astype('float32')
                
                # Generate predictions
                for model_name in models:
                    if model_name in st.session_state.model_results:
                        st.subheader(f"{model_name} Prediction")
                        result = st.session_state.model_results[model_name]
                        if model_name == "Ridge Regression":
                            model_path = result.get('model_path')
                            model = joblib.load(model_path)
                            X_spec = spec_df[st.session_state.feature_cols]
                            predictions_log = model.predict(X_spec)
                            predictions = np.expm1(predictions_log).clip(0)
                            spec_df['predicted_sales'] = predictions
                        else:
                            train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                                    (st.session_state.train_set['family'] == family)]
                            if not train_group.empty:
                                if model_name == "Naive":
                                    last_sale = train_group['sales'].iloc[-1]
                                    predictions = np.full(len(spec_df), last_sale)
                                elif model_name == "Seasonal Naive":
                                    last_week = train_group['sales'].tail(7).values if len(train_group) >= 7 else np.full(7, train_group['sales'].mean())
                                    predictions = np.tile(last_week, (len(spec_df) // 7) + 1)[:len(spec_df)]
                                elif model_name == "Moving Average":
                                    window = min(7, len(train_group))
                                    ma_value = train_group['sales'].tail(window).mean()
                                    predictions = np.full(len(spec_df), ma_value)
                            else:
                                predictions = np.zeros(len(spec_df))
                            spec_df['predicted_sales'] = predictions
                        
                        # Aggregate predictions based on granularity
                        if time_granularity == "Day":
                            predicted_sales = spec_df['predicted_sales'].iloc[0]
                            st.write(f"Predicted Sales for {target_date}: **{predicted_sales:.2f}**")
                        else:
                            avg_sales = spec_df['predicted_sales'].mean()
                            st.write(f"Average Predicted Sales for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                        
                        # Plot
                        plt.figure(figsize=(10, 5))
                        plt.plot(spec_df['date'], spec_df['predicted_sales'], label=f'{model_name} Prediction')
                        plt.title(f"{model_name} Sales Prediction for Store {store_nbr}, Family {family}")
                        plt.xlabel("Date")
                        plt.ylabel("Predicted Sales")
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plot_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_spec_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        # Display plot
                        image = Image.open(plot_path)
                        st.image(image, caption=f"{model_name} Prediction", use_column_width=True)
                    else:
                        st.write(f"Model {model_name} not trained.")
    else:
        st.write("Please upload data and generate predictions in the Training tab.")

# Forecasting Tab
with forecasting_tab:
    st.header("Forecast Sales for Specific Period")
    
    if st.session_state.train_set is not None:
        # Get store numbers and families
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        # User inputs
        st.subheader("Specify Forecasting Details")
        store_nbr = st.selectbox("Select Store Number", store_nbrs, key="forecast_store")
        family = st.selectbox("Select Product Family", families, key="forecast_family")
        onpromotion = st.selectbox("On Promotion?", [0, 1], key="forecast_promo")
        time_granularity = st.selectbox("Select Time Granularity", ["Day", "Month", "Year"], key="forecast_time")
        
        if time_granularity == "Day":
            target_date = st.date_input("Select Date", min_value=datetime(2017, 8, 16), key="forecast_date")
            target_dates = [pd.to_datetime(target_date)]
        elif time_granularity == "Month":
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="forecast_year")
            month = st.number_input("Select Month", min_value=1, max_value=12, value=8, key="forecast_month")
            target_dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-28", freq='D')
        else:  # Year
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="forecast_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        forecast_button = st.button("Generate Forecast")
        
        if forecast_button:
            with st.spinner("Generating forecast..."):
                # Prepare data for forecasting
                forecast_data = []
                for date in target_dates:
                    forecast_data.append({
                        'store_nbr': store_nbr,
                        'family': family,
                        'date': date,
                        'onpromotion': onpromotion,
                        'is_train': 0
                    })
                forecast_df = pd.DataFrame(forecast_data)
                
                # Add features
                forecast_df['day'] = forecast_df['date'].dt.day.astype('int8')
                forecast_df['dow'] = forecast_df['date'].dt.dayofweek.astype('int8')
                forecast_df['month'] = forecast_df['date'].dt.month.astype('int8')
                forecast_df['year'] = forecast_df['date'].dt.year.astype('int16')
                forecast_df['sin_month'] = np.sin(2 * np.pi * forecast_df['month'] / 12).astype('float32')
                forecast_df['cos_month'] = np.cos(2 * np.pi * forecast_df['month'] / 12).astype('float32')
                forecast_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(forecast_df)).astype('int8')
                forecast_df['family_encoded'] = st.session_state.le_family.transform([family] * len(forecast_df)).astype('int8')
                
                # Compute lags and rolling means
                train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                        (st.session_state.train_set['family'] == family)]
                combined = pd.concat([train_group, forecast_df]).sort_values(['store_nbr', 'family', 'date'])
                for lag in [7, 14, 28]:
                    combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(lag).fillna(0).astype('float32')
                combined['roll_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).mean().fillna(0).astype('float32')
                combined['roll_std_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7, min_periods=1).std().fillna(0).astype('float32')
                forecast_df = combined[combined['date'].isin(target_dates)]
                
                # Scale features
                forecast_df[st.session_state.feature_cols] = st.session_state.scaler.transform(forecast_df[st.session_state.feature_cols]).astype('float32')
                
                # Generate forecasts
                for model_name in models:
                    if model_name in st.session_state.model_results:
                        st.subheader(f"{model_name} Forecast")
                        result = st.session_state.model_results[model_name]
                        if model_name == "Ridge Regression":
                            model_path = result.get('model_path')
                            model = joblib.load(model_path)
                            X_forecast = forecast_df[st.session_state.feature_cols]
                            predictions_log = model.predict(X_forecast)
                            predictions = np.expm1(predictions_log).clip(0)
                            forecast_df['predicted_sales'] = predictions
                        else:
                            train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                                    (st.session_state.train_set['family'] == family)]
                            if not train_group.empty:
                                if model_name == "Naive":
                                    last_sale = train_group['sales'].iloc[-1]
                                    predictions = np.full(len(forecast_df), last_sale)
                                elif model_name == "Seasonal Naive":
                                    last_week = train_group['sales'].tail(7).values if len(train_group) >= 7 else np.full(7, train_group['sales'].mean())
                                    predictions = np.tile(last_week, (len(forecast_df) // 7) + 1)[:len(forecast_df)]
                                elif model_name == "Moving Average":
                                    window = min(7, len(train_group))
                                    ma_value = train_group['sales'].tail(window).mean()
                                    predictions = np.full(len(forecast_df), ma_value)
                            else:
                                predictions = np.zeros(len(forecast_df))
                            forecast_df['predicted_sales'] = predictions
                        
                        # Aggregate forecasts based on granularity
                        if time_granularity == "Day":
                            predicted_sales = forecast_df['predicted_sales'].iloc[0]
                            st.write(f"Forecasted Sales for {target_date}: **{predicted_sales:.2f}**")
                        else:
                            avg_sales = forecast_df['predicted_sales'].mean()
                            st.write(f"Average Forecasted Sales for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                        
                        # Plot
                        plt.figure(figsize=(10, 5))
                        plt.plot(forecast_df['date'], forecast_df['predicted_sales'], label=f'{model_name} Forecast')
                        plt.title(f"{model_name} Sales Forecast for Store {store_nbr}, Family {family}")
                        plt.xlabel("Date")
                        plt.ylabel("Predicted Sales")
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plot_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_forecast.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        # Display plot
                        image = Image.open(plot_path)
                        st.image(image, caption=f"{model_name} Forecast", use_column_width=True)
                    else:
                        st.write(f"Model {model_name} not trained.")
    else:
        st.write("Please upload data and generate predictions in the Training tab.")
