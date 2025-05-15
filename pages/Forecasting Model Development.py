import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Constants
MAX_TRAIN_ROWS = 100000
MIN_SAMPLES = 10
MODELS = ["XGBoost", "ARIMA", "SARIMA", "Prophet", "LSTM"]

# Streamlit config
st.set_page_config(layout="wide")
st.title("Sales Forecasting Dashboard")

# Initialize session state
for key, default in {
    "processed_data": None,
    "train": None,
    "test": None,
    "feature_cols": None,
    "scaler": None,
    "le_store": None,
    "le_family": None,
    "date_column": "date",
    "target_column": "sales",
    "predictions": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def load_and_process_data(train_file, test_file, date_col="date", target_col="sales"):
    if not train_file or not test_file:
        st.error("Upload both train and test files.")
        return [None] * 6

    try:
        # Preview structure
        train_sample = pd.read_csv(train_file, nrows=1)
        test_sample = pd.read_csv(test_file, nrows=1)
        st.write(f"Train columns: {train_sample.columns.tolist()}")
        st.write(f"Test columns: {test_sample.columns.tolist()}")

        if date_col not in train_sample or date_col not in test_sample:
            st.error(f"Missing date column '{date_col}'.")
            return [None] * 6
        if target_col not in train_sample:
            st.error(f"Missing target column '{target_col}' in train file.")
            return [None] * 6

        dtypes = {
            "store_nbr": "int32", "family": "category", "onpromotion": "int32",
            target_col: "float32", "id": "int32", "city": "category",
            "state": "category", "type_x": "category", "cluster": "int32",
            "transactions": "int32", "type_y": "category", "locale": "category",
            "locale_name": "category", "description": "category",
            "transferred": "bool", "dcoilwtico": "float32"
        }

        train = pd.read_csv(train_file, dtype=dtypes, parse_dates=[date_col])
        test = pd.read_csv(test_file, dtype=dtypes, parse_dates=[date_col])
        train.drop(columns=[col for col in train if col.startswith("Unnamed")], inplace=True, errors="ignore")
        test.drop(columns=[col for col in test if col.startswith("Unnamed")], inplace=True, errors="ignore")

        if train[date_col].isna().all() or test[date_col].isna().all():
            st.error("No valid date values.")
            return [None] * 6

        if not pd.to_numeric(train[target_col], errors="coerce").notna().all():
            st.error("Target column must be numeric.")
            return [None] * 6

        if len(train) > MAX_TRAIN_ROWS:
            train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

        for df in [train, test]:
            df["month"] = df[date_col].dt.month.astype("int8")
            df["day"] = df[date_col].dt.day.astype("int8")
            df["dow"] = df[date_col].dt.dayofweek.astype("int8")
            df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")

        le_store = LabelEncoder()
        le_family = LabelEncoder()
        train["store_nbr_encoded"] = le_store.fit_transform(train["store_nbr"])
        test["store_nbr_encoded"] = le_store.transform(test["store_nbr"])
        train["family_encoded"] = le_family.fit_transform(train["family"])
        test["family_encoded"] = le_family.transform(test["family"])

        for col in ["city", "state", "locale", "type_x", "type_y"]:
            if col in train and col in test:
                le = LabelEncoder()
                train[f"{col}_encoded"] = le.fit_transform(train[col])
                test[f"{col}_encoded"] = le.transform(test[col])

        train.sort_values([date_col], inplace=True)
        test.sort_values([date_col], inplace=True)

        for lag in [7, 14]:
            train[f"lag_{lag}"] = train.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0)
            test[f"lag_{lag}"] = 0.0

        base_features = ["store_nbr_encoded", "family_encoded", "onpromotion",
                         "month", "day", "dow", "is_weekend",
                         "lag_7", "lag_14", "dcoilwtico", "transactions"]

        cat_encoded = [f"{c}_encoded" for c in ["city", "state", "locale", "type_x", "type_y"] if f"{c}_encoded" in train]
        feature_cols = base_features + cat_encoded

        scaler = StandardScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols])
        test[feature_cols] = scaler.transform(test[feature_cols])

        return train, test, feature_cols, scaler, le_store, le_family

    except Exception as e:
        st.error(f"Error: {e}")
        return [None] * 6
