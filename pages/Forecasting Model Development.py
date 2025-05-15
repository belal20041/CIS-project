import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from statsmodels.tsa.vector_ar.var_model import VAR
import joblib
import tempfile

st.set_page_config(layout="wide")
st.title("Sales Forecasting")

# Initialize session state
if 'forecasting_initialized' not in st.session_state:
    st.session_state.forecasting_initialized = True
    st.session_state.model_results = {}
    st.session_state.train_set = None
    st.session_state.val_set = None
    st.session_state.test = None
    st.session_state.feature_cols = None
    st.session_state.scaler = None
    st.session_state.le_store = None
    st.session_state.le_family = None

# Define tabs
training_tab, prediction_tab, specific_prediction_tab, forecasting_tab = st.tabs(
    ["Training", "Prediction", "Specific Date Prediction", "Forecasting"]
)

# Constants
TRAIN_END = '2017-07-15'
VAL_END = '2017-08-15'
MAX_GROUPS = 50
MAX_MODELS = 1
MAX_TRAIN_ROWS = 150000
MIN_SAMPLES = 14

with training_tab:
    st.header("Train Forecasting Models")
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")
    
    models = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LightGBM", "ETS", 
              "TBATS", "Holt-Winters", "VAR", "Random Forest"]
    selected_models = st.multiselect("Select Models to Train", models, 
                                    default=["ARIMA"], max_selections=MAX_MODELS)
    
    train_button = st.button("Generate Predictions")
    
    if train_button and train_file and test_file and selected_models:
        with st.spinner("Processing data..."):
            def load_and_process_data(train_file, test_file):
                # Define dtypes
                train_dtypes = {'store_nbr': 'int32', 'family': 'category', 
                               'sales': 'float32', 'onpromotion': 'int32'}
                test_dtypes = {'store_nbr': 'int32', 'family': 'category', 
                              'onpromotion': 'int32', 'id': 'int32'}
                
                # Read train.csv in chunks for sampling
                chunksize = 50000
                store_family_counts = {}
                train_content = train_file.getvalue()
                train_chunks = pd.read_csv(BytesIO(train_content), dtype=train_dtypes, chunksize=chunksize)
                for chunk in train_chunks:
                    for (store, family), group in chunk.groupby(['store_nbr', 'family']):
                        store_family_counts[(store, family)] = store_family_counts.get(
                            (store, family), 0) + len(group)
                
                # Stratified sampling
                num_pairs = len(store_family_counts)
                rows_per_pair = max(1, MAX_TRAIN_ROWS // num_pairs)
                train_samples = []
                train_chunks = pd.read_csv(BytesIO(train_content), dtype=train_dtypes, chunksize=chunksize)
                for chunk in train_chunks:
                    sampled_chunk = chunk.groupby(['store_nbr', 'family']).apply(
                        lambda x: x.sample(n=min(len(x), rows_per_pair), random_state=42)
                    ).reset_index(drop=True)
                    train_samples.append(sampled_chunk)
                
                train = pd.concat(train_samples, ignore_index=True)
                if len(train) > MAX_TRAIN_ROWS:
                    train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)
                
                # Read test.csv
                test_content = test_file.getvalue()
                test = pd.read_csv(BytesIO(test_content), dtype=test_dtypes)
                
                # Attempt to parse dates with multiple formats
                date_formats = [
                    ('%Y-%m-%d', 'YYYY-MM-DD'),
                    ('%d-%m-%Y', 'DD-MM-YYYY'),
                    ('%d/%m/%Y', 'DD/MM/YYYY')
                ]
                
                # Parse train dates
                train_date_parsed = False
                for fmt, fmt_name in date_formats:
                    train['date'] = pd.to_datetime(train['date'], format=fmt, errors='coerce')
                    if not train['date'].isna().any():
                        train_date_parsed = True
                        break
                if not train_date_parsed:
                    st.error("Invalid date formats in train.csv. Tried YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY. Please ensure dates are in one of these formats.")
                    return None, None, None, None, None, None, None
                
                # Parse test dates
                test_date_parsed = False
                for fmt, fmt_name in date_formats:
                    test['date'] = pd.to_datetime(test['date'], format=fmt, errors='coerce')
                    if not test['date'].isna().any():
                        test_date_parsed = True
                        break
                if not test_date_parsed:
                    st.error("Invalid date formats in test.csv. Tried YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY. Please ensure dates are in one of these formats.")
                    return None, None, None, None, None, None, None
                
                # Combine train and test
                train['is_train'] = 1
                test['is_train'] = 0
                combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', 'date'])
                
                # Aggregate by store, family, date
                agg_dict = {'sales': 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
                combined = combined.groupby(['store_nbr', 'family', 'date']).agg(agg_dict).reset_index()
                combined = combined.astype({'store_nbr': 'int32', 'family': 'category', 
                                          'date': 'datetime64[ns]', 'sales': 'float32', 
                                          'onpromotion': 'int32', 'is_train': 'int8'})
                
                # Limit to MAX_GROUPS
                store_family_pairs = combined[['store_nbr', 'family']].drop_duplicates()
                if len(store_family_pairs) > MAX_GROUPS:
                    store_family_pairs = store_family_pairs.sample(n=MAX_GROUPS, random_state=42)
                    combined = combined.merge(store_family_pairs, on=['store_nbr', 'family'])
                
                # Fill missing dates
                date_range = pd.date_range(start=combined['date'].min(), end=combined['date'].max(), freq='D')
                index = pd.MultiIndex.from_product(
                    [store_family_pairs['store_nbr'], store_family_pairs['family'], date_range],
                    names=['store_nbr', 'family', 'date']
                )
                combined = combined.set_index(['store_nbr', 'family', 'date']).reindex(index).reset_index()
                combined['sales'] = combined['sales'].fillna(0).astype('float32')
                combined['onpromotion'] = combined['onpromotion'].fillna(0).astype('int32')
                combined['is_train'] = combined['is_train'].fillna(0).astype('int8')
                
                # Feature engineering
                combined['day'] = combined['date'].dt.day.astype('int8')
                combined['dow'] = combined['date'].dt.dayofweek.astype('int8')
                combined['month'] = combined['date'].dt.month.astype('int8')
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                
                # Encode categorical variables
                le_store = LabelEncoder()
                le_family = LabelEncoder()
                combined['store_nbr_encoded'] = le_store.fit_transform(combined['store_nbr']).astype('int8')
                combined['family_encoded'] = le_family.fit_transform(combined['family']).astype('int8')
                
                # Scale features
                feature_cols = ['onpromotion', 'day', 'dow', 'month', 'store_nbr_encoded', 'family_encoded', 'lag_7']
                scaler = StandardScaler()
                combined[feature_cols] = scaler.fit_transform(combined[feature_cols]).astype('float32')
                
                # Split into train, validation, test
                train = combined[combined['is_train'] == 1]
                test = combined[combined['is_train'] == 0].drop(['sales'], axis=1)
                train_set = train[train['date'] <= TRAIN_END]
                val_set = train[(train['date'] > TRAIN_END) & (train['date'] <= VAL_END)]
                
                return train_set, val_set, test, feature_cols, scaler, le_store, le_family
            
            def clipped_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mask = y_true > 0
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
            
            # Process data
            result = load_and_process_data(train_file, test_file)
            if result[0] is None:  # Check if date validation failed
                st.stop()
            train_set, val_set, test, feature_cols, scaler, le_store, le_family = result
            
            # Store in session state
            st.session_state.train_set = train_set
            st.session_state.val_set = val_set
            st.session_state.test = test
            st.session_state.feature_cols = feature_cols
            st.session_state.scaler = scaler
            st.session_state.le_store = le_store
            st.session_state.le_family = le_family
            
            # Train models
            for model_name in selected_models:
                st.write(f"Generating predictions for {model_name}...")
                temp_dir = tempfile.gettempdir()
                pred_dict = {}
                
                train_pairs = set(train_set[['store_nbr', 'family']].drop_duplicates().itertuples(index=False, name=None))
                group_iter = [(pair, group) for pair, group in val_set.groupby(['store_nbr', 'family']) if pair in train_pairs]
                if len(group_iter) > MAX_GROUPS:
                    group_iter = group_iter[:MAX_GROUPS]
                
                for (store, family), group in group_iter:
                    train_group = train_set[(train_set['store_nbr'] == store) & (train_set['family'] == family)]
                    val_group = group.sort_values('date')
                    dates = val_group['date'].values
                    actuals = val_group['sales'].values
                    preds = np.zeros(len(actuals))
                    
                    if len(train_group) >= MIN_SAMPLES and train_group['sales'].var() > 0:
                        if model_name == "ARIMA":
                            model = ARIMA(train_group['sales'], order=(3,1,0))
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "SARIMA":
                            model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                            fit = model.fit(disp=False)
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "Prophet":
                            df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                            model = Prophet(daily_seasonality=True)
                            model.fit(df)
                            future = pd.DataFrame({'ds': val_group['date']})
                            forecast = model.predict(future)
                            preds = forecast['yhat'].values
                        
                        elif model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                            X_train = train_group[feature_cols]
                            y_train = np.log1p(train_group['sales'].clip(0))
                            X_val = val_group[feature_cols]
                            if model_name == "XGBoost":
                                model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                            elif model_name == "LightGBM":
                                model = LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                            else:
                                model = RandomForestRegressor(n_estimators=50, random_state=42)
                            model.fit(X_train, y_train)
                            preds_log = model.predict(X_val)
                            preds = np.expm1(preds_log).clip(0)
                            joblib.dump(model, f"{temp_dir}/{model_name.lower()}_{store}_{family}.pt")
                        
                        elif model_name == "ETS":
                            model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "TBATS":
                            model = TBATS(seasonal_periods=[7], use_box_cox=False)
                            fit = model.fit(train_group['sales'])
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "Holt-Winters":
                            model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "VAR":
                            var_data = train_group[['sales', 'onpromotion']]
                            model = VAR(var_data)
                            fit = model.fit(maxlags=7)
                            lag_order = fit.k_ar
                            last_obs = var_data.values[-lag_order:]
                            forecast = fit.forecast(last_obs, steps=len(actuals))
                            preds = forecast[:, 0].clip(0)
                    
                    preds = np.clip(preds, 0, None)
                    pred_dict[(store, family)] = {'dates': dates, 'actuals': actuals, 'preds': preds}
                
                # Calculate metrics
                all_actuals = []
                all_preds = []
                for data in pred_dict.values():
                    all_actuals.extend(data['actuals'])
                    all_preds.extend(data['preds'])
                
                metrics = {
                    'rmsle': np.sqrt(mean_squared_error(np.log1p(all_actuals), np.log1p(all_preds))),
                    'rmse': np.sqrt(mean_squared_error(all_actuals, all_preds)),
                    'mae': mean_absolute_error(all_actuals, all_preds),
                    'mape': clipped_mape(all_actuals, all_preds)
                }
                
                # Plot predictions
                n_plot = min(100, len(all_actuals))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates[:n_plot], y=all_actuals[:n_plot], mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=dates[:n_plot], y=all_preds[:n_plot], mode='lines', name='Predicted', line=dict(color='orange')))
                fig.update_layout(
                    title=f"{model_name} Predictions",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    xaxis_tickangle=45,
                    yaxis_gridcolor='lightgray'
                )
                st.plotly_chart(fig)
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'metrics': metrics,
                    'pred_dict': pred_dict
                }
                
                # Display metrics
                st.write(f"### {model_name} Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                col3.metric("MAE", f"{metrics['mae']:.4f}")
                col4.metric("MAPE (%)", f"{metrics['mape']:.2f}")
            
            st.success("Predictions generated!")

with prediction_tab:
    st.header("Visualize Predictions")
    
    if st.session_state.train_set is not None:
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        st.subheader("Visualize Predictions vs Actual")
        store_nbr = st.selectbox("Select Store Number", store_nbrs, key="viz_store")
        family = st.selectbox("Select Product Family", families, key="viz_family")
        selected_models = st.multiselect("Select Models to Visualize", models, 
                                        default=["ARIMA"], key="viz_models")
        
        if selected_models:
            for model_name in selected_models:
                st.subheader(f"{model_name} Predictions")
                if model_name in st.session_state.model_results:
                    result = st.session_state.model_results[model_name]
                    metrics = result['metrics']
                    pred_dict = result['pred_dict']
                    
                    key = (store_nbr, family)
                    if key in pred_dict:
                        data = pred_dict[key]
                        dates = data['dates']
                        actual = data['actuals']
                        pred = data['preds']
                        
                        n_plot = min(100, len(actual))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dates[:n_plot], y=actual[:n_plot], mode='lines', name='Actual', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dates[:n_plot], y=pred[:n_plot], mode='lines', name='Predicted', line=dict(color='orange')))
                        fig.update_layout(
                            title=f"{model_name} Predictions: Store {store_nbr}, Family {family}",
                            xaxis_title="Date",
                            yaxis_title="Sales",
                            xaxis_tickangle=45,
                            yaxis_gridcolor='lightgray'
                        )
                        st.plotly_chart(fig)
                        
                        st.write("### Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                        col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                        col3.metric("MAE", f"{metrics['mae']:.4f}")
                        col4.metric("MAPE (%)", f"{metrics['mape']:.2f}")

with specific_prediction_tab:
    st.header("Predict Sales for Specific Date")
    
    if st.session_state.train_set is not None:
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
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
        else:
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="spec_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        predict_button = st.button("Predict Sales")
        
        if predict_button:
            with st.spinner("Generating prediction..."):
                spec_data = [
                    {'store_nbr': store_nbr, 'family': family, 'date': date, 'onpromotion': onpromotion, 'is_train': 0}
                    for date in target_dates
                ]
                spec_df = pd.DataFrame(spec_data)
                
                spec_df['day'] = spec_df['date'].dt.day.astype('int8')
                spec_df['dow'] = spec_df['date'].dt.dayofweek.astype('int8')
                spec_df['month'] = spec_df['date'].dt.month.astype('int8')
                spec_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(spec_df)).astype('int8')
                spec_df['family_encoded'] = st.session_state.le_family.transform([family] * len(spec_df)).astype('int8')
                
                train_group = st.session_state.train_set[
                    (st.session_state.train_set['store_nbr'] == store_nbr) & 
                    (st.session_state.train_set['family'] == family)
                ]
                combined = pd.concat([train_group, spec_df]).sort_values(['store_nbr', 'family', 'date'])
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                spec_df = combined[combined['date'].isin(target_dates)]
                
                spec_df[st.session_state.feature_cols] = st.session_state.scaler.transform(
                    spec_df[st.session_state.feature_cols]
                ).astype('float32')
                
                for model_name in selected_models:
                    st.subheader(f"{model_name} Prediction")
                    if model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                        model_path = f"{tempfile.gettempdir()}/{model_name.lower()}_{store_nbr}_{family}.pt"
                        model = joblib.load(model_path)
                        X_spec = spec_df[st.session_state.feature_cols]
                        predictions_log = model.predict(X_spec)
                        predictions = np.expm1(predictions_log).clip(0)
                        spec_df['predicted_sales'] = predictions
                    else:
                        if len(train_group) >= MIN_SAMPLES and train_group['sales'].var() > 0:
                            if model_name == "ARIMA":
                                model = ARIMA(train_group['sales'], order=(3,1,0))
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "SARIMA":
                                model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                                fit = model.fit(disp=False)
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "Prophet":
                                df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                                model = Prophet(daily_seasonality=True)
                                model.fit(df)
                                future = pd.DataFrame({'ds': spec_df['date']})
                                forecast = model.predict(future)
                                predictions = forecast['yhat'].values
                            elif model_name == "ETS":
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "TBATS":
                                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                                fit = model.fit(train_group['sales'])
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "Holt-Winters":
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "VAR":
                                var_data = train_group[['sales', 'onpromotion']]
                                model = VAR(var_data)
                                fit = model.fit(maxlags=7)
                                lag_order = fit.k_ar
                                last_obs = var_data.values[-lag_order:]
                                forecast = fit.forecast(last_obs, steps=len(spec_df))
                                predictions = forecast[:, 0]
                            spec_df['predicted_sales'] = np.clip(predictions, 0, None)
                
                    if time_granularity == "Day":
                        predicted_sales = spec_df['predicted_sales'].iloc[0]
                        st.write(f"Predicted Sales for {target_date}: **{predicted_sales:.2f}**")
                    else:
                        avg_sales = spec_df['predicted_sales'].mean()
                        st.write(f"Average Predicted Sales for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=spec_df['date'], y=spec_df['predicted_sales'], mode='lines', name=f'{model_name} Prediction'))
                    fig.update_layout(
                        title=f"{model_name} Sales Prediction for Store {store_nbr}, Family {family}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Sales",
                        xaxis_tickangle=45,
                        yaxis_gridcolor='lightgray'
                    )
                    st.plotly_chart(fig)

with forecasting_tab:
    st.header("Forecast Sales for Specific Period")
    
    if st.session_state.train_set is not None:
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
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
        else:
            year = st.number_input("Select Year", min_value=2017, max_value=2030, value=2017, key="forecast_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        forecast_button = st.button("Generate Forecast")
        
        if forecast_button:
            with st.spinner("Generating forecast..."):
                forecast_data = [
                    {'store_nbr': store_nbr, 'family': family, 'date': date, 'onpromotion': onpromotion, 'is_train': 0}
                    for date in target_dates
                ]
                forecast_df = pd.DataFrame(forecast_data)
                
                forecast_df['day'] = forecast_df['date'].dt.day.astype('int8')
                forecast_df['dow'] = forecast_df['date'].dt.dayofweek.astype('int8')
                forecast_df['month'] = forecast_df['date'].dt.month.astype('int8')
                forecast_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(forecast_df)).astype('int8')
                forecast_df['family_encoded'] = st.session_state.le_family.transform([family] * len(forecast_df)).astype('int8')
                
                train_group = st.session_state.train_set[
                    (st.session_state.train_set['store_nbr'] == store_nbr) & 
                    (st.session_state.train_set['family'] == family)
                ]
                combined = pd.concat([train_group, forecast_df]).sort_values(['store_nbr', 'family', 'date'])
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                forecast_df = combined[combined['date'].isin(target_dates)]
                
                forecast_df[st.session_state.feature_cols] = st.session_state.scaler.transform(
                    forecast_df[st.session_state.feature_cols]
                ).astype('float32')
                
                for model_name in selected_models:
                    st.subheader(f"{model_name} Forecast")
                    if model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                        model_path = f"{tempfile.gettempdir()}/{model_name.lower()}_{store_nbr}_{family}.pt"
                        model = joblib.load(model_path)
                        X_forecast = forecast_df[st.session_state.feature_cols]
                        predictions_log = model.predict(X_forecast)
                        predictions = np.expm1(predictions_log).clip(0)
                        forecast_df['predicted_sales'] = predictions
                    else:
                        if len(train_group) >= MIN_SAMPLES and train_group['sales'].var() > 0:
                            if model_name == "ARIMA":
                                model = ARIMA(train_group['sales'], order=(3,1,0))
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "SARIMA":
                                model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                                fit = model.fit(disp=False)
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "Prophet":
                                df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                                model = Prophet(daily_seasonality=True)
                                model.fit(df)
                                future = pd.DataFrame({'ds': forecast_df['date']})
                                forecast = model.predict(future)
                                predictions = forecast['yhat'].values
                            elif model_name == "ETS":
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "TBATS":
                                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                                fit = model.fit(train_group['sales'])
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "Holt-Winters":
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "VAR":
                                var_data = train_group[['sales', 'onpromotion']]
                                model = VAR(var_data)
                                fit = model.fit(maxlags=7)
                                lag_order = fit.k_ar
                                last_obs = var_data.values[-lag_order:]
                                forecast = fit.forecast(last_obs, steps=len(forecast_df))
                                predictions = forecast[:, 0]
                            forecast_df['predicted_sales'] = np.clip(predictions, 0, None)
                
                    if time_granularity == "Day":
                        predicted_sales = forecast_df['predicted_sales'].iloc[0]
                        st.write(f"Forecasted Sales for {target_date}: **{predicted_sales:.2f}**")
                    else:
                        avg_sales = forecast_df['predicted_sales'].mean()
                        st.write(f"Average Forecasted Sales for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_sales'], mode='lines', name=f'{model_name} Forecast'))
                    fig.update_layout(
                        title=f"{model_name} Sales Forecast for Store {store_nbr}, Family {family}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Sales",
                        xaxis_tickangle=45,
                        yaxis_gridcolor='lightgray'
                    )
                    st.plotly_chart(fig)
