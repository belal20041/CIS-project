import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from statsmodels.tsa.vector_ar.var_model import VAR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import tempfile

# Constants
TRAIN_END = '2017-07-15'
VAL_END = '2017-08-15'
MAX_GROUPS = 50
MAX_MODELS = 1
MAX_TRAIN_ROWS = 150000
MIN_SAMPLES = 14
MODELS = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LightGBM", "ETS", 
          "TBATS", "Holt-Winters", "VAR", "Random Forest"]

# Streamlit configuration
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
    st.session_state.date_column = 'date'
    st.session_state.target_column = 'sales'

# Utility functions
def to_csv_download(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

def clipped_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0

# Data processing
def load_and_process_data(train_file, test_file, date_col, target_col):
    # Validate files
    if not train_file or not test_file:
        st.error("Please upload both train and test CSV files.")
        return None, None, None, None, None

    train_content = train_file.getvalue()
    test_content = test_file.getvalue()
    if not train_content or not test_content:
        st.error("One of the CSV files is empty.")
        return None, None, None, None, None

    # Define dtypes
    train_dtypes = {'store_nbr': 'int32', 'family': 'category', 'onpromotion': 'int32', target_col: 'float32'}
    test_dtypes = {'store_nbr': 'int32', 'family': 'category', 'onpromotion': 'int32', 'id': 'int32'}
    optional_cols = ['city', 'state', 'type_x', 'cluster', 'transactions', 'dcoilwtico', 
                     'locale', 'locale_name', 'description', 'transferred', 'type_y']

    try:
        train_cols = pd.read_csv(BytesIO(train_content), nrows=1).columns
        test_cols = pd.read_csv(BytesIO(test_content), nrows=1).columns
    except pd.errors.EmptyDataError:
        st.error("One of the CSV files is empty or invalid.")
        return None, None, None, None, None

    for col in optional_cols:
        if col in train_cols:
            train_dtypes[col] = ('category' if col in ['city', 'state', 'type_x', 'locale', 
                                                     'locale_name', 'description', 'type_y'] 
                                else 'float32' if col in ['dcoilwtico', 'transactions', 'cluster'] 
                                else 'bool')
        if col in test_cols:
            test_dtypes[col] = ('category' if col in ['city', 'state', 'type_x', 'locale', 
                                                    'locale_name', 'description', 'type_y'] 
                               else 'float32' if col in ['dcoilwtico', 'transactions', 'cluster'] 
                               else 'bool')

    # Read and sample train data
    chunksize = 50000
    store_family_counts = {}
    train_chunks = pd.read_csv(BytesIO(train_content), dtype=train_dtypes, chunksize=chunksize)
    for chunk in train_chunks:
        # Drop Unnamed: 17 if present
        if 'Unnamed: 17' in chunk.columns:
            chunk = chunk.drop(columns=['Unnamed: 17'])
        if not pd.to_numeric(chunk[target_col], errors='coerce').notna().all():
            st.error(f"Target column '{target_col}' contains non-numeric values.")
            return None, None, None, None, None
        for (store, family), group in chunk.groupby(['store_nbr', 'family']):
            store_family_counts[(store, family)] = store_family_counts.get((store, family), 0) + len(group)

    num_pairs = len(store_family_counts)
    rows_per_pair = max(1, MAX_TRAIN_ROWS // num_pairs)
    train_samples = []
    train_chunks = pd.read_csv(BytesIO(train_content), dtype=train_dtypes, chunksize=chunksize)
    for chunk in train_chunks:
        # Drop Unnamed: 17 if present
        if 'Unnamed: 17' in chunk.columns:
            chunk = chunk.drop(columns=['Unnamed: 17'])
        sampled_chunk = chunk.groupby(['store_nbr', 'family']).apply(
            lambda x: x.sample(n=min(len(x), rows_per_pair), random_state=42)
        ).reset_index(drop=True)
        train_samples.append(sampled_chunk)

    train = pd.concat(train_samples, ignore_index=True)
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

    # Read test data
    try:
        test = pd.read_csv(BytesIO(test_content), dtype=test_dtypes)
        # Drop Unnamed: 17 if present
        if 'Unnamed: 17' in test.columns:
            test = test.drop(columns=['Unnamed: 17'])
    except pd.errors.EmptyDataError:
        st.error("test.csv is empty or invalid.")
        return None, None, None, None, None

    # Validate and parse dates
    train[date_col] = pd.to_datetime(train[date_col], errors='coerce')
    test[date_col] = pd.to_datetime(test[date_col], errors='coerce')
    if train[date_col].isna().any():
        st.error(f"Invalid date formats in train.csv column '{date_col}'. Sample invalid values: {train[train[date_col].isna()][date_col].head().tolist()}")
        return None, None, None, None, None
    if test[date_col].isna().any():
        st.error(f"Invalid date formats in test.csv column '{date_col}'. Sample invalid values: {test[test[date_col].isna()][date_col].head().tolist()}")
        return None, None, None, None, None

    # Feature engineering
    train['sales_onpromo'] = train[target_col] * train['onpromotion']
    for col, group_col in [('city', 'city'), ('state', 'state')]:
        if col in train.columns and col in test.columns:
            means = train.groupby(group_col)[target_col].mean().to_dict()
            train[f'{col}_encoded'] = train[col].map(means).astype('float32')
            test[f'{col}_encoded'] = test[col].map(means).astype('float32').fillna(0)

    for df in [train, test]:
        df['month'] = df[date_col].dt.month.astype('int8')
        df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype('int8')

    train['avg_sales_store_month'] = train.groupby(['store_nbr', 'month'])[target_col].transform('mean').astype('float32')
    if 'store_nbr' in test.columns:
        store_month_means = train.groupby(['store_nbr', 'month'])[target_col].mean().to_dict()
        test['avg_sales_store_month'] = test.apply(
            lambda x: store_month_means.get((x['store_nbr'], x['month']), 0), axis=1
        ).astype('float32')

    for col, freq_col in [('family', 'family'), ('locale_name', 'locale_name')]:
        if col in train.columns and col in test.columns:
            freq = train[freq_col].value_counts(normalize=True).to_dict()
            train[f'{col}_encoded'] = train[col].map(freq).astype('float32')
            test[f'{col}_encoded'] = test[col].map(freq).fillna(0).astype('float32')

    if 'description' in train.columns and 'description' in test.columns:
        train['is_holiday'] = train['description'].str.contains('Holiday|Navidad', case=False, na=False).astype('int8')
        test['is_holiday'] = test['description'].str.contains('Holiday|Navidad', case=False, na=False).astype('int8')

    for col, mapping in [('locale', {'National': 2, 'Regional': 1, 'Local': 0}),
                         ('type_y', {'Holiday': 2, 'Event': 1, 'Bridge': 0})]:
        if col in train.columns and col in test.columns:
            train[f'{col}_encoded'] = train[col].map(mapping).fillna(0).astype('int8')
            test[f'{col}_encoded'] = test[col].map(mapping).fillna(0).astype('int8')

    if 'dcoilwtico' in train.columns and 'dcoilwtico' in test.columns:
        q25, q75 = train['dcoilwtico'].quantile([0.25, 0.75])
        bins = [-np.inf, q25, np.inf] if q25 == q75 else [-np.inf, q25, q75, np.inf]
        labels = ['low', 'high'] if q25 == q75 else ['low', 'medium', 'high']
        train['dcoilwtico_bin'] = pd.cut(train['dcoilwtico'], bins=bins, labels=labels).astype('category')
        test['dcoilwtico_bin'] = pd.cut(test['dcoilwtico'], bins=bins, labels=labels).astype('category')
        bin_order = {label: idx for idx, label in enumerate(labels)}
        train['dcoilwtico_bin_encoded'] = train['dcoilwtico_bin'].map(bin_order).fillna(0).astype('int8')
        test['dcoilwtico_bin_encoded'] = test['dcoilwtico_bin'].map(bin_order).fillna(0).astype('int8')

    train['season'] = pd.cut(train['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4']).astype('category')
    test['season'] = pd.cut(test['month'], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4']).astype('category')
    season_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    train['season_encoded'] = train['season'].map(season_order).astype('int8')
    test['season_encoded'] = test['season'].map(season_order).astype('int8')

    train['promo_weekend'] = train['onpromotion'] * train['is_weekend'].astype('int32')
    test['promo_weekend'] = test['onpromotion'] * test['is_weekend'].astype('int32')

    # Combine and aggregate
    train['is_train'] = 1
    test['is_train'] = 0
    combined = pd.concat([train, test]).sort_values(['store_nbr', 'family', date_col])
    agg_dict = {target_col: 'mean', 'onpromotion': 'sum', 'is_train': 'first', 'id': 'first',
                'sales_onpromo': 'mean', 'city_encoded': 'mean', 'state_encoded': 'mean',
                'avg_sales_store_month': 'mean', 'family_encoded': 'mean', 'locale_name_encoded': 'mean',
                'is_holiday': 'max', 'locale_encoded': 'max', 'type_y_encoded': 'max',
                'transferred': 'max', 'is_weekend': 'max', 'dcoilwtico_bin_encoded': 'max',
                'season_encoded': 'max', 'promo_weekend': 'sum'}
    for col in ['city', 'state', 'type_x', 'cluster', 'transactions', 'dcoilwtico']:
        if col in combined.columns:
            agg_dict[col] = 'first' if col in ['city', 'state', 'type_x'] else 'mean'

    combined = combined.groupby(['store_nbr', 'family', date_col]).agg(agg_dict).reset_index()
    combined = combined.astype({'store_nbr': 'int32', 'family': 'category', date_col: 'datetime64[ns]',
                               target_col: 'float32', 'onpromotion': 'int32', 'is_train': 'int8',
                               'sales_onpromo': 'float32', 'city_encoded': 'float32', 'state_encoded': 'float32',
                               'avg_sales_store_month': 'float32', 'family_encoded': 'float32',
                               'locale_name_encoded': 'float32', 'is_holiday': 'int8', 'locale_encoded': 'int8',
                               'type_y_encoded': 'int8', 'transferred': 'int8', 'is_weekend': 'int8',
                               'dcoilwtico_bin_encoded': 'int8', 'season_encoded': 'int8', 'promo_weekend': 'int32'})

    # Limit groups
    store_family_pairs = combined[['store_nbr', 'family']].drop_duplicates()
    if len(store_family_pairs) > MAX_GROUPS:
        store_family_pairs = store_family_pairs.sample(n=MAX_GROUPS, random_state=42)
        combined = combined.merge(store_family_pairs, on=['store_nbr', 'family'])

    # Fill missing dates
    date_range = pd.date_range(start=combined[date_col].min(), end=combined[date_col].max(), freq='D')
    index = pd.MultiIndex.from_product(
        [store_family_pairs['store_nbr'], store_family_pairs['family'], date_range],
        names=['store_nbr', 'family', date_col]
    )
    combined = combined.set_index(['store_nbr', 'family', date_col]).reindex(index).reset_index()
    combined[target_col] = combined[target_col].fillna(0).astype('float32')
    combined['onpromotion'] = combined['onpromotion'].fillna(0).astype('int32')
    combined['is_train'] = combined['is_train'].fillna(0).astype('int8')
    for col in ['sales_onpromo', 'city_encoded', 'state_encoded', 'avg_sales_store_month',
                'family_encoded', 'locale_name_encoded', 'dcoilwtico', 'transactions', 'cluster', 'promo_weekend']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0).astype('float32' if col != 'promo_weekend' else 'int32')
    for col in ['is_holiday', 'locale_encoded', 'type_y_encoded', 'transferred', 'is_weekend', 
                'dcoilwtico_bin_encoded', 'season_encoded']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0).astype('int8')
    for col in ['city', 'state', 'type_x']:
        if col in combined.columns:
            combined[col] = combined[col].fillna(combined[col].mode().iloc[0]).astype('category')

    # Additional features
    combined['day'] = combined[date_col].dt.day.astype('int8')
    combined['dow'] = combined[date_col].dt.dayofweek.astype('int8')
    combined['month'] = combined[date_col].dt.month.astype('int8')
    combined['quarter'] = combined[date_col].dt.quarter.astype('int8')
    combined['year'] = combined[date_col].dt.year.astype('int16')

    for lag in [7, 14, 28]:
        combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(lag).fillna(0).astype('float32')
    for window in [7, 14]:
        combined[f'roll_mean_{window}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0).astype('float32')
        combined[f'roll_std_{window}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).astype('float32')

    combined['onpromotion_binary'] = (combined['onpromotion'] > 0).astype('int8')
    combined['lag_promo_7'] = combined.groupby(['store_nbr', 'family'])['onpromotion'].shift(7).fillna(0).astype('int32')
    for col in ['transactions', 'dcoilwtico']:
        if col in combined.columns:
            combined[f'{col}_lag_7'] = combined.groupby(['store_nbr', 'family'])[col].shift(7).fillna(0).astype('float32')

    # Encode categoricals
    le_store = LabelEncoder()
    le_family = LabelEncoder()
    combined['store_nbr_encoded'] = le_store.fit_transform(combined['store_nbr']).astype('int8')
    combined['family_encoded_le'] = le_family.fit_transform(combined['family']).astype('int8')
    for col in ['city', 'state', 'type_x']:
        if col in combined.columns:
            le = LabelEncoder()
            combined[f'{col}_encoded_le'] = le.fit_transform(combined[col]).astype('int8')

    # Scale features
    feature_cols = [
        'onpromotion', 'onpromotion_binary', 'lag_promo_7', 'day', 'dow', 'month', 'quarter', 'year',
        'is_weekend', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_std_7', 'roll_mean_14', 'roll_std_14',
        'store_nbr_encoded', 'family_encoded_le', 'sales_onpromo', 'city_encoded', 'state_encoded',
        'avg_sales_store_month', 'family_encoded', 'locale_name_encoded', 'is_holiday', 'locale_encoded',
        'type_y_encoded', 'transferred', 'dcoilwtico_bin_encoded', 'season_encoded', 'promo_weekend'
    ]
    for col in ['city_encoded_le', 'state_encoded_le', 'type_x_encoded_le', 'transactions_lag_7', 'dcoilwtico_lag_7']:
        if col in combined.columns:
            feature_cols.append(col)

    scaler = StandardScaler()
    combined[feature_cols] = scaler.fit_transform(combined[feature_cols]).astype('float32')

    return combined, feature_cols, scaler, le_store, le_family
# Modeling functions
def train_model(model_name, train_set, val_set, feature_cols, date_col, target_col, max_groups, min_samples):
    temp_dir = tempfile.gettempdir()
    pred_dict = {}
    
    train_pairs = set(train_set[['store_nbr', 'family']].drop_duplicates().itertuples(index=False, name=None))
    group_iter = [(pair, group) for pair, group in val_set.groupby(['store_nbr', 'family']) if pair in train_pairs]
    if len(group_iter) > max_groups:
        group_iter = group_iter[:max_groups]
    
    for (store, family), group in group_iter:
        train_group = train_set[(train_set['store_nbr'] == store) & (train_set['family'] == family)]
        val_group = group.sort_values(date_col)
        dates = val_group[date_col].values
        actuals = val_group[target_col].values
        preds = np.zeros(len(actuals))
        
        if len(train_group) >= min_samples and train_group[target_col].var() > 0:
            if model_name == "ARIMA":
                model = ARIMA(train_group[target_col], order=(3,1,0))
                fit = model.fit()
                preds = fit.forecast(steps=len(actuals))
            elif model_name == "SARIMA":
                model = SARIMAX(train_group[target_col], order=(1,1,1), seasonal_order=(1,1,1,7))
                fit = model.fit(disp=False)
                preds = fit.forecast(steps=len(actuals))
            elif model_name == "Prophet":
                df = train_group[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
                model = Prophet(daily_seasonality=True)
                model.fit(df)
                future = pd.DataFrame({'ds': val_group[date_col]})
                forecast = model.predict(future)
                preds = forecast['yhat'].values
            elif model_name in ["XGBoost", "LightGBM", "Random Forest"]:
                X_train = train_group[feature_cols]
                y_train = np.log1p(train_group[target_col].clip(0))
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
                model = ExponentialSmoothing(train_group[target_col], trend='add', seasonal='add', seasonal_periods=7)
                fit = model.fit()
                preds = fit.forecast(steps=len(actuals))
            elif model_name == "TBATS":
                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                fit = model.fit(train_group[target_col])
                preds = fit.forecast(steps=len(actuals))
            elif model_name == "Holt-Winters":
                model = ExponentialSmoothing(train_group[target_col], trend='add', seasonal='mul', seasonal_periods=7)
                fit = model.fit()
                preds = fit.forecast(steps=len(actuals))
            elif model_name == "VAR":
                var_data = train_group[[target_col, 'onpromotion']]
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
    
    return metrics, pred_dict

def predict_specific_date(model_name, train_set, store_nbr, family, onpromotion, target_dates, 
                         feature_cols, scaler, le_store, le_family, date_col, target_col, min_samples):
    spec_data = [
        {'store_nbr': store_nbr, 'family': family, date_col: date, 'onpromotion': onpromotion, 'is_train': 0}
        for date in target_dates
    ]
    spec_df = pd.DataFrame(spec_data)
    
    spec_df['day'] = spec_df[date_col].dt.day.astype('int8')
    spec_df['dow'] = spec_df[date_col].dt.dayofweek.astype('int8')
    spec_df['month'] = spec_df[date_col].dt.month.astype('int8')
    spec_df['quarter'] = spec_df[date_col].dt.quarter.astype('int8')
    spec_df['year'] = spec_df[date_col].dt.year.astype('int16')
    spec_df['is_weekend'] = spec_df['dow'].isin([5, 6]).astype('int8')
    
    spec_df['store_nbr_encoded'] = le_store.transform([store_nbr] * len(spec_df)).astype('int8')
    spec_df['family_encoded_le'] = le_family.transform([family] * len(spec_df)).astype('int8')
    
    spec_df['onpromotion_binary'] = (spec_df['onpromotion'] > 0).astype('int8')
    spec_df['lag_promo_7'] = 0
    spec_df['promo_weekend'] = spec_df['onpromotion'] * spec_df['is_weekend'].astype('int32')
    
    if 'city' in train_set.columns:
        city_means = train_set.groupby('city')[target_col].mean().to_dict()
        spec_df['city_encoded'] = city_means.get(train_set['city'].mode().iloc[0], 0)
    if 'state' in train_set.columns:
        state_means = train_set.groupby('state')[target_col].mean().to_dict()
        spec_df['state_encoded'] = state_means.get(train_set['state'].mode().iloc[0], 0)
    if 'family' in train_set.columns:
        family_freq = train_set['family'].value_counts(normalize=True).to_dict()
        spec_df['family_encoded'] = family_freq.get(family, 0)
    spec_df['avg_sales_store_month'] = train_set[train_set['store_nbr'] == store_nbr].groupby('month')[target_col].mean().get(spec_df['month'].iloc[0], 0)
    spec_df['locale_name_encoded'] = 0
    spec_df['is_holiday'] = 0
    spec_df['locale_encoded'] = 0
    spec_df['type_y_encoded'] = 0
    spec_df['transferred'] = 0
    spec_df['dcoilwtico_bin_encoded'] = 0
    spec_df['season_encoded'] = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}.get(
        pd.cut([spec_df['month'].iloc[0]], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])[0], 0
    )
    spec_df['sales_onpromo'] = 0
    
    train_group = train_set[(train_set['store_nbr'] == store_nbr) & (train_set['family'] == family)]
    combined = pd.concat([train_group, spec_df]).sort_values(['store_nbr', 'family', date_col])
    
    for lag in [7, 14, 28]:
        combined[f'lag_{lag}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(lag).fillna(0).astype('float32')
    for window in [7, 14]:
        combined[f'roll_mean_{window}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(window=window, min_periods=1).mean().fillna(0).astype('float32')
        combined[f'roll_std_{window}'] = combined.groupby(['store_nbr', 'family'])[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).astype('float32')
    
    spec_df = combined[combined[date_col].isin(target_dates)]
    spec_df[feature_cols] = scaler.transform(spec_df[feature_cols]).astype('float32')
    
    predictions = np.zeros(len(spec_df))
    if len(train_group) >= min_samples and train_group[target_col].var() > 0:
        if model_name in ["XGBoost", "LightGBM", "Random Forest"]:
            model_path = f"{tempfile.gettempdir()}/{model_name.lower()}_{store_nbr}_{family}.pt"
            try:
                model = joblib.load(model_path)
                X_spec = spec_df[feature_cols]
                predictions_log = model.predict(X_spec)
                predictions = np.expm1(predictions_log).clip(0)
            except FileNotFoundError:
                st.error(f"Model file for {model_name} not found.")
                return predictions
        else:
            if model_name == "ARIMA":
                model = ARIMA(train_group[target_col], order=(3,1,0))
                fit = model.fit()
                predictions = fit.forecast(steps=len(spec_df))
            elif model_name == "SARIMA":
                model = SARIMAX(train_group[target_col], order=(1,1,1), seasonal_order=(1,1,1,7))
                fit = model.fit(disp=False)
                predictions = fit.forecast(steps=len(spec_df))
            elif model_name == "Prophet":
                df = train_group[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
                model = Prophet(daily_seasonality=True)
                model.fit(df)
                future = pd.DataFrame({'ds': spec_df[date_col]})
                forecast = model.predict(future)
                predictions = forecast['yhat'].values
            elif model_name == "ETS":
                model = ExponentialSmoothing(train_group[target_col], trend='add', seasonal='add', seasonal_periods=7)
                fit = model.fit()
                predictions = fit.forecast(steps=len(spec_df))
            elif model_name == "TBATS":
                model = TBATS(seasonal_periods=[7], use_box_cox=False)
                fit = model.fit(train_group[target_col])
                predictions = fit.forecast(steps=len(spec_df))
            elif model_name == "Holt-Winters":
                model = ExponentialSmoothing(train_group[target_col], trend='add', seasonal='mul', seasonal_periods=7)
                fit = model.fit()
                predictions = fit.forecast(steps=len(spec_df))
            elif model_name == "VAR":
                var_data = train_group[[target_col, 'onpromotion']]
                model = VAR(var_data)
                fit = model.fit(maxlags=7)
                lag_order = fit.k_ar
                last_obs = var_data.values[-lag_order:]
                forecast = fit.forecast(last_obs, steps=len(spec_df))
                predictions = forecast[:, 0]
    
    return np.clip(predictions, 0, None)

def forecast_period(model_name, train_set, store_nbr, family, onpromotion, target_dates, 
                    feature_cols, scaler, le_store, le_family, date_col, target_col, min_samples):
    return predict_specific_date(model_name, train_set, store_nbr, family, onpromotion, target_dates, 
                                feature_cols, scaler, le_store, le_family, date_col, target_col, min_samples)

# Streamlit UI
training_tab, prediction_tab, specific_prediction_tab, forecasting_tab = st.tabs(
    ["Training", "Prediction", "Specific Date Prediction", "Forecasting"]
)

with training_tab:
    st.header("Train Forecasting Models")
    
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")
    
    date_column = None
    target_column = None
    if train_file and test_file:
        try:
            train_df = pd.read_csv(train_file, nrows=1)
            columns = train_df.columns.tolist()
            
            st.subheader("Select Columns")
            date_column = st.selectbox("Select Date Column", columns, 
                                     index=columns.index('date') if 'date' in columns else 0,
                                     key="date_column_select")
            target_column = st.selectbox("Select Target Column (Numeric)", columns,
                                       index=columns.index('sales') if 'sales' in columns else 0,
                                       key="target_column_select")
            
            st.session_state.date_column = date_column
            st.session_state.target_column = target_column
        except Exception as e:
            st.error(f"Error reading train.csv: {str(e)}")
            st.stop()
    
    selected_models = st.multiselect("Select Models to Train", MODELS, 
                                    default=["ARIMA"], max_selections=MAX_MODELS)
    train_button = st.button("Generate Predictions")
    
    if train_button and train_file and test_file and selected_models and date_column and target_column:
        with st.spinner("Processing data..."):
            result = load_and_process_data(train_file, test_file, date_column, target_column)
            if result[0] is None:
                st.stop()
            combined, feature_cols, scaler, le_store, le_family = result
            
            st.subheader("Processed Data Preview")
            st.dataframe(combined.head(100))
            st.write("### Data Summary")
            st.write(f"Total rows: {len(combined)}")
            st.write(f"Unique dates: {combined[date_column].nunique()}")
            st.write(f"Date range: {combined[date_column].min()} to {combined[date_column].max()}")
            st.write(f"Null counts:\n{combined.isna().sum()}")
            
            st.download_button(
                label="Download Processed Data as CSV",
                data=to_csv_download(combined),
                file_name="processed_data.csv",
                mime="text/csv"
            )
            
            train_set, val_set, test = split_data(combined, date_column, target_column)
            st.session_state.train_set = train_set
            st.session_state.val_set = val_set
            st.session_state.test = test
            st.session_state.feature_cols = feature_cols
            st.session_state.scaler = scaler
            st.session_state.le_store = le_store
            st.session_state.le_family = le_family
            
            for model_name in selected_models:
                st.write(f"Training {model_name}...")
                metrics, pred_dict = train_model(
                    model_name, train_set, val_set, feature_cols, 
                    date_column, target_column, MAX_GROUPS, MIN_SAMPLES
                )
                
                all_actuals = []
                all_preds = []
                all_dates = []
                for data in pred_dict.values():
                    all_actuals.extend(data['actuals'])
                    all_preds.extend(data['preds'])
                    all_dates.extend(data['dates'])
                
                n_plot = min(100, len(all_actuals))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=all_dates[:n_plot], y=all_actuals[:n_plot], 
                                       mode='lines', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=all_dates[:n_plot], y=all_preds[:n_plot], 
                                       mode='lines', name='Predicted', line=dict(color='orange')))
                fig.update_layout(
                    title=f"{model_name} Predictions",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    xaxis_tickangle=45,
                    yaxis_gridcolor='lightgray'
                )
                st.plotly_chart(fig)
                
                st.write(f"### {model_name} Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSLE", f"{metrics['rmsle']:.4f}")
                col2.metric("RMSE", f"{metrics['rmse']:.4f}")
                col3.metric("MAE", f"{metrics['mae']:.4f}")
                col4.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                
                st.session_state.model_results[model_name] = {
                    'metrics': metrics,
                    'pred_dict': pred_dict
                }
            
            st.success("Training completed!")

with prediction_tab:
    st.header("Visualize Predictions")
    
    if st.session_state.train_set is not None:
        store_nbrs = sorted(st.session_state.train_set['store_nbr'].unique())
        families = sorted(st.session_state.train_set['family'].unique())
        
        st.subheader("Select Parameters")
        store_nbr = st.selectbox("Store Number", store_nbrs, key="viz_store")
        family = st.selectbox("Product Family", families, key="viz_family")
        selected_models = st.multiselect("Models", MODELS, default=["ARIMA"], key="viz_models")
        
        if selected_models:
            for model_name in selected_models:
                if model_name in st.session_state.model_results:
                    st.subheader(f"{model_name} Predictions")
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
                        fig.add_trace(go.Scatter(x=dates[:n_plot], y=actual[:n_plot], 
                                               mode='lines', name='Actual', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=dates[:n_plot], y=pred[:n_plot], 
                                               mode='lines', name='Predicted', line=dict(color='orange')))
                        fig.update_layout(
                            title=f"{model_name} Predictions: Store {store_nbr}, Family {family}",
                            xaxis_title="Date",
                            yaxis_title=st.session_state.target_column,
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
        store_nbr = st.selectbox("Store Number", store_nbrs, key="spec_store")
        family = st.selectbox("Product Family", families, key="spec_family")
        onpromotion = st.selectbox("On Promotion?", [0, 1], key="spec_promo")
        time_granularity = st.selectbox("Time Granularity", ["Day", "Month", "Year"], key="spec_time")
        
        if time_granularity == "Day":
            target_date = st.date_input("Date", min_value=datetime(2017, 8, 16), key="spec_date")
            target_dates = [pd.to_datetime(target_date)]
        elif time_granularity == "Month":
            year = st.number_input("Year", min_value=2017, max_value=2030, value=2017, key="spec_year")
            month = st.number_input("Month", min_value=1, max_value=12, value=8, key="spec_month")
            target_dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-28", freq='D')
        else:
            year = st.number_input("Year", min_value=2017, max_value=2030, value=2017, key="spec_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        predict_button = st.button("Predict Sales")
        
        if predict_button:
            with st.spinner("Generating prediction..."):
                for model_name in selected_models:
                    predictions = predict_specific_date(
                        model_name, st.session_state.train_set, store_nbr, family, 
                        onpromotion, target_dates, st.session_state.feature_cols,
                        st.session_state.scaler, st.session_state.le_store, 
                        st.session_state.le_family, st.session_state.date_column,
                        st.session_state.target_column, MIN_SAMPLES
                    )
                    
                    st.subheader(f"{model_name} Prediction")
                    if time_granularity == "Day":
                        predicted_sales = predictions[0]
                        st.write(f"Predicted {st.session_state.target_column} for {target_date}: **{predicted_sales:.2f}**")
                    else:
                        avg_sales = np.mean(predictions)
                        st.write(f"Average Predicted {st.session_state.target_column} for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_dates, y=predictions, mode='lines', 
                                           name=f'{model_name} Prediction'))
                    fig.update_layout(
                        title=f"{model_name} Prediction for Store {store_nbr}, Family {family}",
                        xaxis_title="Date",
                        yaxis_title=f"Predicted {st.session_state.target_column}",
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
        store_nbr = st.selectbox("Store Number", store_nbrs, key="forecast_store")
        family = st.selectbox("Product Family", families, key="forecast_family")
        onpromotion = st.selectbox("On Promotion?", [0, 1], key="forecast_promo")
        time_granularity = st.selectbox("Time Granularity", ["Day", "Month", "Year"], key="forecast_time")
        
        if time_granularity == "Day":
            target_date = st.date_input("Date", min_value=datetime(2017, 8, 16), key="forecast_date")
            target_dates = [pd.to_datetime(target_date)]
        elif time_granularity == "Month":
            year = st.number_input("Year", min_value=2017, max_value=2030, value=2017, key="forecast_year")
            month = st.number_input("Month", min_value=1, max_value=12, value=8, key="forecast_month")
            target_dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-28", freq='D')
        else:
            year = st.number_input("Year", min_value=2017, max_value=2030, value=2017, key="forecast_year")
            target_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        forecast_button = st.button("Generate Forecast")
        
        if forecast_button:
            with st.spinner("Generating forecast..."):
                for model_name in selected_models:
                    predictions = forecast_period(
                        model_name, st.session_state.train_set, store_nbr, family, 
                        onpromotion, target_dates, st.session_state.feature_cols,
                        st.session_state.scaler, st.session_state.le_store, 
                        st.session_state.le_family, st.session_state.date_column,
                        st.session_state.target_column, MIN_SAMPLES
                    )
                    
                    st.subheader(f"{model_name} Forecast")
                    if time_granularity == "Day":
                        predicted_sales = predictions[0]
                        st.write(f"Forecasted {st.session_state.target_column} for {target_date}: **{predicted_sales:.2f}**")
                    else:
                        avg_sales = np.mean(predictions)
                        st.write(f"Average Forecasted {st.session_state.target_column} for {time_granularity} ({year}-{month:02d} if Month): **{avg_sales:.2f}**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=target_dates, y=predictions, mode='lines', 
                                           name=f'{model_name} Forecast'))
                    fig.update_layout(
                        title=f"{model_name} Forecast for Store {store_nbr}, Family {family}",
                        xaxis_title="Date",
                        yaxis_title=f"Predicted {st.session_state.target_column}",
                        xaxis_tickangle=45,
                        yaxis_gridcolor='lightgray'
                    )
                    st.plotly_chart(fig)
