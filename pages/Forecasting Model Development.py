import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Constants
MAX_TRAIN_ROWS = 100_000
MIN_TRAIN_SAMPLES = 10
MODELS = ["XGBoost", "ARIMA", "SARIMA", "Prophet", "LSTM"]
REQUIRED_COLUMNS = ["id", "date", "store_nbr", "family", "sales", "onpromotion"]

# Streamlit Setup
st.set_page_config(layout="wide")
st.title("Sales Forecasting Dashboard")

# Session State Initialization
if "state" not in st.session_state:
    st.session_state.state = {
        "train_data": None,
        "test_data": None,
        "features": None,
        "scaler": None,
        "label_encoders": {},
        "date_col": "date",
        "target_col": "sales",
        "predictions": {},
    }

def load_data(train_file, test_file, date_col, target_col, delimiter=","):
    if not train_file or not test_file:
        st.error("Upload both train and test CSV files.")
        return None, None, None, None, None

    # Preview data
    train_file.seek(0)
    test_file.seek(0)
    train_preview = train_file.read().decode("utf-8").splitlines()[:5]
    test_preview = test_file.read().decode("utf-8").splitlines()[:5]
    train_file.seek(0)
    test_file.seek(0)

    st.write("Train CSV Preview (first 5 rows):")
    for line in train_preview:
        st.text(line)
    st.write("Test CSV Preview (first 5 rows):")
    for line in test_preview:
        st.text(line)

    # Load small sample to validate columns
    train_sample = pd.read_csv(train_file, nrows=1, delimiter=delimiter)
    train_file.seek(0)
    test_sample = pd.read_csv(test_file, nrows=1, delimiter=delimiter)
    test_file.seek(0)

    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in train_sample.columns]
    if missing_cols:
        st.error(f"Train CSV missing required columns: {missing_cols}")
        return None, None, None, None, None

    if date_col not in train_sample.columns or date_col not in test_sample.columns:
        st.error(f"Date column '{date_col}' not found in both datasets.")
        return None, None, None, None, None

    if target_col not in train_sample.columns:
        st.error(f"Target column '{target_col}' not found in train CSV.")
        return None, None, None, None, None

    # Data types to optimize memory
    dtypes = {
        "store_nbr": "int32",
        "family": "category",
        "onpromotion": "int32",
        "id": "int32",
        "city": "category",
        "state": "category",
        "type_x": "category",
        "cluster": "int32",
        "transactions": "int32",
        "type_y": "category",
        "locale": "category",
        "locale_name": "category",
        "description": "category",
        "transferred": "bool",
        "dcoilwtico": "float32",
        target_col: "float32",
    }

    # Load full datasets
    train = pd.read_csv(train_file, dtype=dtypes, parse_dates=[date_col], delimiter=delimiter)
    test = pd.read_csv(test_file, dtype=dtypes, parse_dates=[date_col], delimiter=delimiter)

    # Clean unwanted columns
    train = train.loc[:, ~train.columns.str.startswith("Unnamed")]
    test = test.loc[:, ~test.columns.str.startswith("Unnamed")]

    # Validate date parsing
    if train[date_col].isna().all() or test[date_col].isna().all():
        st.error(f"Date column '{date_col}' contains no valid dates.")
        return None, None, None, None, None

    # Check target numeric
    if not pd.to_numeric(train[target_col], errors="coerce").notna().all():
        st.error(f"Target column '{target_col}' contains non-numeric values.")
        return None, None, None, None, None

    # Limit training rows for performance
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

    # Feature Engineering: date-based
    for df in [train, test]:
        df["month"] = df[date_col].dt.month.astype("int8")
        df["day"] = df[date_col].dt.day.astype("int8")
        df["weekday"] = df[date_col].dt.dayofweek.astype("int8")
        df["is_weekend"] = df["weekday"].isin([5,6]).astype("int8")

    # Encode categorical features
    le_store = LabelEncoder()
    le_family = LabelEncoder()
    train["store_enc"] = le_store.fit_transform(train["store_nbr"]).astype("int8")
    test["store_enc"] = le_store.transform(test["store_nbr"]).astype("int8")
    train["family_enc"] = le_family.fit_transform(train["family"]).astype("int8")
    test["family_enc"] = le_family.transform(test["family"]).astype("int8")

    # Encode extra categorical columns if present
    extra_cats = ["city", "state", "locale", "type_x", "type_y"]
    for col in extra_cats:
        if col in train.columns and col in test.columns:
            le = LabelEncoder()
            train[f"{col}_enc"] = le.fit_transform(train[col]).astype("int8")
            test[f"{col}_enc"] = le.transform(test[col]).astype("int8")

    # Lag features for time series context
    train = train.sort_values(date_col)
    test = test.sort_values(date_col)
    for lag in [7, 14]:
        train[f"lag_{lag}"] = train.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0).astype("float32")
        # For test, fill lag with 0 as future sales unknown
        test[f"lag_{lag}"] = 0.0

    # Define features for modeling
    features = [
        "store_enc", "family_enc", "onpromotion", "month", "day",
        "weekday", "is_weekend", "lag_7", "lag_14", "dcoilwtico", "transactions"
    ]
    for col in [f"{c}_enc" for c in extra_cats]:
        if col in train.columns:
            features.append(col)

    # Scale numeric features
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    return train, test, features, scaler, le_store, le_family

def prepare_lstm_sequences(df, target_col, features, seq_len=14):
    X, y = [], []
    data = df[features + [target_col]].values
    for i in range(len(df) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

def train_predict_model(train, test, features, date_col, target_col, model_name):
    predictions = []
    test = test.sort_values(["store_nbr", "family", date_col])

    for (store, family), group in test.groupby(["store_nbr", "family"]):
        train_sub = train[(train["store_nbr"] == store) & (train["family"] == family)]
        test_sub = group.sort_values(date_col)

        if len(train_sub) < MIN_TRAIN_SAMPLES or train_sub[target_col].var() == 0:
            preds = np.zeros(len(test_sub))
        else:
            if model_name == "XGBoost":
                X_train = train_sub[features]
                y_train = np.log1p(train_sub[target_col].clip(0))
                X_test = test_sub[features]
                model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
                preds = np.expm1(model.predict(X_test)).clip(0)
            elif model_name == "ARIMA":
                model = ARIMA(train_sub[target_col], order=(3,1,0))
                fit = model.fit()
                preds = fit.forecast(steps=len(test_sub)).clip(0)
            elif model_name == "SARIMA":
                model = SARIMAX(train_sub[target_col], order=(1,1,1), seasonal_order=(1,1,1,7))
                fit = model.fit(disp=False)
                preds = fit.forecast(steps=len(test_sub)).clip(0)
            elif model_name == "Prophet":
                df_prophet = train_sub[[date_col, target_col]].rename(columns={date_col:"ds", target_col:"y"})
                model = Prophet(daily_seasonality=True)
                model.fit(df_prophet)
                            future = test_sub[[date_col]].rename(columns={date_col:"ds"})
            forecast = model.predict(future)
            preds = forecast["yhat"].clip(0)
        elif model_name == "LSTM":
            seq_len = 14
            X_train, y_train = prepare_lstm_sequences(train_sub, target_col, features, seq_len)
            if len(X_train) == 0:
                preds = np.zeros(len(test_sub))
            else:
                model = Sequential([
                    LSTM(50, activation="relu", input_shape=(seq_len, len(features))),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X_train, y_train, epochs=5, verbose=0)
                # For test, use last seq_len rows from train_sub features
                X_test_seq = train_sub[features].tail(seq_len).values.reshape(1, seq_len, len(features))
                preds = []
                for _ in range(len(test_sub)):
                    pred = model.predict(X_test_seq)[0,0]
                    preds.append(pred)
                    # Slide window with predicted value (ignore for features)
                preds = np.array(preds).clip(0)
        else:
            preds = np.zeros(len(test_sub))

    predictions.append(pd.DataFrame({
        "id": test_sub["id"],
        "predicted_sales": preds
    }))

return pd.concat(predictions).sort_values("id")
