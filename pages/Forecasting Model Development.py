import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

st.title("Sales Forecasting Dashboard")

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_results = {}
    st.session_state.train_set = None
    st.session_state.val_set = None
    st.session_state.test = None
    st.session_state.sub = None
    st.session_state.feature_cols = None
    st.session_state.scaler = None
    st.session_state.le_store = None
    st.session_state.le_family = None

training_tab, prediction_tab, specific_prediction_tab, forecasting_tab = st.tabs(["Training", "Prediction", "Specific Date Prediction", "Forecasting"])

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
    sub_file = st.file_uploader("Upload Submission CSV", type="csv", key="uploader_sub")
    
    models = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LightGBM", "ETS", "TBATS", "Holt-Winters", "VAR", "Random Forest"]
    selected_models = st.multiselect("Select Models to Train", models, default=["ARIMA"], max_selections=MAX_MODELS)
    
    train_button = st.button("Generate Predictions")
    
    if train_button and train_file and test_file and sub_file and selected_models:
        with st.spinner("Processing data..."):
            def load_and_process_data(train_file, test_file, sub_file):
                train_dtypes = {'store_nbr': 'int32', 'family': 'category', 'sales': 'float32', 'onpromotion': 'int32'}
                test_dtypes = {'store_nbr': 'int32', 'family': 'category', 'onpromotion': 'int32', 'id': 'int32'}
                
                chunksize = 50000
                target_rows = MAX_TRAIN_ROWS
                train_samples = []
                total_rows = 0
                
                # Read train.csv in chunks
                train_chunks = pd.read_csv(train_file, dtype=train_dtypes, chunksize=chunksize)
                store_family_counts = {}
                for chunk in train_chunks:
                    total_rows += len(chunk)
                    for (store, family), group in chunk.groupby(['store_nbr', 'family']):
                        store_family_counts[(store, family)] = store_family_counts.get((store, family), 0) + len(group)
                
                num_pairs = len(store_family_counts)
                rows_per_pair = max(1, target_rows // num_pairs)
                
                # Sample train.csv
                train_chunks = pd.read_csv(train_file, dtype=train_dtypes, chunksize=chunksize)
                for chunk in train_chunks:
                    sampled_chunk = chunk.groupby(['store_nbr', 'family']).apply(
                        lambda x: x.sample(n=min(len(x), rows_per_pair), random_state=42)
                    ).reset_index(drop=True)
                    train_samples.append(sampled_chunk)
                
                train = pd.concat(train_samples, ignore_index=True)
                if len(train) > target_rows:
                    train = train.sample(n=target_rows, random_state=42)
                
                # Read test.csv and sample_submission.csv
                test = pd.read_csv(test_file, dtype=test_dtypes)
                sub = pd.read_csv(sub_file, dtype={'id': 'int32', 'sales': 'float32'})
                
                train['date'] = pd.to_datetime(train['date'], dayfirst=True)
                test['date'] = pd.to_datetime(test['date'], dayfirst=True)
                
                train['is_train'] = 1
                test['is_train'] = 0
                combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', 'date'])
                agg_dict = {'sales': 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first'}
                combined = combined.groupby(['store_nbr', 'family', 'date']).agg(agg_dict).reset_index()
                combined = combined.astype({'store_nbr': 'int32', 'family': 'category', 'date': 'datetime64[ns]', 
                                           'sales': 'float32', 'onpromotion': 'int32', 'is_train': 'int8'})
                
                store_families = combined[['store_nbr', 'family']].drop_duplicates()
                if len(store_families) > MAX_GROUPS:
                    store_families = store_families.sample(n=MAX_GROUPS, random_state=42)
                    combined = combined.merge(store_families, on=['store_nbr', 'family'])
                
                date_range = pd.date_range(start=combined['date'].min(), end=combined['date'].max(), freq='D')
                index = pd.MultiIndex.from_product([store_families['store_nbr'], store_families['family'], date_range], 
                                                   names=['store_nbr', 'family', 'date'])
                combined = combined.set_index(['store_nbr', 'family', 'date']).reindex(index).reset_index()
                combined['sales'] = combined['sales'].fillna(0).astype('float32')
                combined['onpromotion'] = combined['onpromotion'].fillna(0).astype('int32')
                combined['is_train'] = combined['is_train'].fillna(0).astype('int8')
                
                combined['day'] = combined['date'].dt.day.astype('int8')
                combined['dow'] = combined['date'].dt.dayofweek.astype('int8')
                combined['month'] = combined['date'].dt.month.astype('int8')
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                
                from sklearn.preprocessing import LabelEncoder
                le_store = LabelEncoder()
                le_family = LabelEncoder()
                combined['store_nbr_encoded'] = le_store.fit_transform(combined['store_nbr']).astype('int8')
                combined['family_encoded'] = le_family.fit_transform(combined['family']).astype('int8')
                
                feature_cols = ['onpromotion', 'day', 'dow', 'month', 'store_nbr_encoded', 'family_encoded', 'lag_7']
                
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                combined[feature_cols] = scaler.fit_transform(combined[feature_cols]).astype('float32')
                
                train = combined[combined['is_train'] == 1]
                test = combined[combined['is_train'] == 0].drop(['sales'], axis=1)
                train_set = train[train['date'] <= TRAIN_END]
                val_set = train[(train['date'] > TRAIN_END) & (train['date'] <= VAL_END)]
                
                return train_set, val_set, test, sub, feature_cols, scaler, le_store, le_family
            
            def clipped_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mask = y_true > 0
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
            
            train_set, val_set, test, sub, feature_cols, scaler, le_store, le_family = load_and_process_data(train_file, test_file, sub_file)
            
            st.session_state.train_set = train_set
            st.session_state.val_set = val_set
            st.session_state.test = test
            st.session_state.sub = sub
            st.session_state.feature_cols = feature_cols
            st.session_state.scaler = scaler
            st.session_state.le_store = le_store
            st.session_state.le_family = le_family
            
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
                            from statsmodels.tsa.arima.model import ARIMA
                            model = ARIMA(train_group['sales'], order=(3,1,0))
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "SARIMA":
                            from statsmodels.tsa.statespace.sarimax import SARIMAX
                            model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                            fit = model.fit(disp=False)
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "Prophet":
                            from prophet import Prophet
                            df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                            model = Prophet(daily_seasonality=True)
                            model.fit(df)
                            future = pd.DataFrame({'ds': val_group['date']})
                            forecast = model.predict(future)
                            preds = forecast['yhat'].values
                        
                        elif model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                            from sklearn.ensemble import RandomForestRegressor
                            import joblib
                            X_train = train_group[feature_cols]
                            y_train = np.log1p(train_group['sales'].clip(0))
                            X_val = val_group[feature_cols]
                            if model_name == "XGBoost":
                                from xgboost import XGBRegressor
                                model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                            elif model_name == "LightGBM":
                                from lightgbm import LGBMRegressor
                                model = LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                            else:
                                model = RandomForestRegressor(n_estimators=50, random_state=42)
                            model.fit(X_train, y_train)
                            preds_log = model.predict(X_val)
                            preds = np.expm1(preds_log).clip(0)
                            joblib.dump(model, os.path.join(temp_dir, f"{model_name.lower()}_{store}_{family}.pt"))
                        
                        elif model_name == "ETS":
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing
                            model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "TBATS":
                            from tbats import TBATS
                            model = TBATS(seasonal_periods=[7], use_box_cox=False)
                            fit = model.fit(train_group['sales'])
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "Holt-Winters":
                            from statsmodels.tsa.holtwinters import ExponentialSmoothing
                            model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                            fit = model.fit()
                            preds = fit.forecast(steps=len(actuals))
                        
                        elif model_name == "VAR":
                            from statsmodels.tsa.vector_ar.var_model import VAR
                            var_data = train_group[['sales', 'onpromotion']]
                            model = VAR(var_data)
                            fit = model.fit(maxlags=7)
                            lag_order = fit.k_ar
                            last_obs = var_data.values[-lag_order:]
                            forecast = fit.forecast(last_obs, steps=len(actuals))
                            preds = forecast[:, 0].clip(0)
                    
                    preds = np.clip(preds, 0, None)
                    pred_dict[(store, family)] = {'dates': dates, 'actuals': actuals, 'preds': preds}
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                all_actuals = []
                all_preds = []
                for key, data in pred_dict.items():
                    all_actuals.extend(data['actuals'])
                    all_preds.extend(data['preds'])
                
                metrics = {
                    'rmsle': np.sqrt(mean_squared_error(np.log1p(all_actuals), np.log1p(all_preds))),
                    'rmse': np.sqrt(mean_squared_error(all_actuals, all_preds)),
                    'mae': mean_absolute_error(all_actuals, all_preds),
                    'mape': clipped_mape(all_actuals, all_preds)
                }
                
                n_plot = min(100, len(all_actuals))
                plt.figure(figsize=(10, 5))
                plt.plot(dates[:n_plot], all_actuals[:n_plot], label='Actual')
                plt.plot(dates[:n_plot], all_preds[:n_plot], label='Predicted')
                plt.title(f"{model_name} Predictions")
                plt.xlabel("Date")
                plt.ylabel("Sales")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(temp_dir, f"{model_name.lower()}_pred.png")
                plt.savefig(plot_path)
                plt.close()
                
                st.session_state.model_results[model_name] = {
                    'metrics': metrics,
                    'plot_path': plot_path,
                    'pred_dict': pred_dict
                }
                
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
        selected_models = st.multiselect("Select Models to Visualize", models, default=["ARIMA"], key="viz_models")
        
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
                        plt.figure(figsize=(10, 5))
                        plt.plot(dates[:n_plot], actual[:n_plot], label='Actual', color='blue')
                        plt.plot(dates[:n_plot], pred[:n_plot], label='Predicted', color='orange')
                        plt.title(f"{model_name} Predictions: Store {store_nbr}, Family {family}")
                        plt.xlabel("Date")
                        plt.ylabel("Sales")
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plot_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_custom_pred.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        image = Image.open(plot_path)
                        st.image(image, caption=f"{model_name} Predictions vs Actual", use_column_width=True)
                        
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
                
                spec_df['day'] = spec_df['date'].dt.day.astype('int8')
                spec_df['dow'] = spec_df['date'].dt.dayofweek.astype('int8')
                spec_df['month'] = spec_df['date'].dt.month.astype('int8')
                spec_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(spec_df)).astype('int8')
                spec_df['family_encoded'] = st.session_state.le_family.transform([family] * len(spec_df)).astype('int8')
                
                train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                        (st.session_state.train_set['family'] == family)]
                combined = pd.concat([train_group, spec_df]).sort_values(['store_nbr', 'family', 'date'])
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                spec_df = combined[combined['date'].isin(target_dates)]
                
                spec_df[st.session_state.feature_cols] = st.session_state.scaler.transform(spec_df[st.session_state.feature_cols]).astype('float32')
                
                for model_name in selected_models:
                    st.subheader(f"{model_name} Prediction")
                    if model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                        import joblib
                        model_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_{store_nbr}_{family}.pt")
                        model = joblib.load(model_path)
                        X_spec = spec_df[st.session_state.feature_cols]
                        predictions_log = model.predict(X_spec)
                        predictions = np.expm1(predictions_log).clip(0)
                        spec_df['predicted_sales'] = predictions
                    else:
                        if len(train_group) >= MIN_SAMPLES and train_group['sales'].var() > 0:
                            if model_name == "ARIMA":
                                from statsmodels.tsa.arima.model import ARIMA
                                model = ARIMA(train_group['sales'], order=(3,1,0))
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "SARIMA":
                                from statsmodels.tsa.statespace.sarimax import SARIMAX
                                model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                                fit = model.fit(disp=False)
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "Prophet":
                                from prophet import Prophet
                                df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                                model = Prophet(daily_seasonality=True)
                                model.fit(df)
                                future = pd.DataFrame({'ds': spec_df['date']})
                                forecast = model.predict(future)
                                predictions = forecast['yhat'].values
                            elif model_name == "ETS":
                                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "TBATS":
                                from tbats import TBATS
                                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                                fit = model.fit(train_group['sales'])
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "Holt-Winters":
                                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(spec_df))
                            elif model_name == "VAR":
                                from statsmodels.tsa.vector_ar.var_model import VAR
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
                    
                    image = Image.open(plot_path)
                    st.image(image, caption=f"{model_name} Prediction", use_column_width=True)

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
                
                forecast_df['day'] = forecast_df['date'].dt.day.astype('int8')
                forecast_df['dow'] = forecast_df['date'].dt.dayofweek.astype('int8')
                forecast_df['month'] = forecast_df['date'].dt.month.astype('int8')
                forecast_df['store_nbr_encoded'] = st.session_state.le_store.transform([store_nbr] * len(forecast_df)).astype('int8')
                forecast_df['family_encoded'] = st.session_state.le_family.transform([family] * len(forecast_df)).astype('int8')
                
                train_group = st.session_state.train_set[(st.session_state.train_set['store_nbr'] == store_nbr) & 
                                                        (st.session_state.train_set['family'] == family)]
                combined = pd.concat([train_group, forecast_df]).sort_values(['store_nbr', 'family', 'date'])
                combined['lag_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(7).fillna(0).astype('float32')
                forecast_df = combined[combined['date'].isin(target_dates)]
                
                forecast_df[st.session_state.feature_cols] = st.session_state.scaler.transform(forecast_df[st.session_state.feature_cols]).astype('float32')
                
                for model_name in selected_models:
                    st.subheader(f"{model_name} Forecast")
                    if model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                        import joblib
                        model_path = os.path.join(tempfile.gettempdir(), f"{model_name.lower()}_{store_nbr}_{family}.pt")
                        model = joblib.load(model_path)
                        X_forecast = forecast_df[st.session_state.feature_cols]
                        predictions_log = model.predict(X_forecast)
                        predictions = np.expm1(predictions_log).clip(0)
                        forecast_df['predicted_sales'] = predictions
                    else:
                        if len(train_group) >= MIN_SAMPLES and train_group['sales'].var() > 0:
                            if model_name == "ARIMA":
                                from statsmodels.tsa.arima.model import ARIMA
                                model = ARIMA(train_group['sales'], order=(3,1,0))
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "SARIMA":
                                from statsmodels.tsa.statespace.sarimax import SARIMAX
                                model = SARIMAX(train_group['sales'], order=(1,1,1), seasonal_order=(1,1,1,7))
                                fit = model.fit(disp=False)
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "Prophet":
                                from prophet import Prophet
                                df = train_group[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
                                model = Prophet(daily_seasonality=True)
                                model.fit(df)
                                future = pd.DataFrame({'ds': forecast_df['date']})
                                forecast = model.predict(future)
                                predictions = forecast['yhat'].values
                            elif model_name == "ETS":
                                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='add', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "TBATS":
                                from tbats import TBATS
                                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                                fit = model.fit(train_group['sales'])
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "Holt-Winters":
                                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                                model = ExponentialSmoothing(train_group['sales'], trend='add', seasonal='mul', seasonal_periods=7)
                                fit = model.fit()
                                predictions = fit.forecast(steps=len(forecast_df))
                            elif model_name == "VAR":
                                from statsmodels.tsa.vector_ar.var_model import VAR
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
                    
                    image = Image.open(plot_path)
                    st.image(image, caption=f"{model_name} Forecast", use_column_width=True)
