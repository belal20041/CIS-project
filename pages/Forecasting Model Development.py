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

# Streamlit configuration
st.set_page_config(layout="wide")
st.title("Sales Forecasting Dashboard")

# Initialize session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.train = None
    st.session_state.test = None
    st.session_state.feature_cols = None
    st.session_state.scaler = None
    st.session_state.le_store = None
    st.session_state.le_family = None
    st.session_state.date_column = "date"
    st.session_state.target_column = "sales"
    st.session_state.predictions = {}

def load_and_process_data(train_file, test_file, date_col="date", target_col="sales"):
    """Load and process train/test CSV files with simplified feature engineering."""
    try:
        if not train_file or not test_file:
            st.error("Please upload both train and test CSV files.")
            return None, None, None, None, None, None

        # Define dtypes
        dtypes = {
            "store_nbr": "int32",
            "family": "category",
            "onpromotion": "int32",
            target_col: "float32",
            "id": "int32"
        }
        optional_cols = ["city", "state", "cluster", "transactions", "dcoilwtico", "locale"]
        for col in optional_cols:
            dtypes[col] = "category" if col in ["city", "state", "locale"] else "float32"

        # Load data
        train = pd.read_csv(train_file, dtype=dtypes, parse_dates=[date_col])
        test = pd.read_csv(test_file, dtype=dtypes, parse_dates=[date_col])

        # Drop unwanted columns
        train = train.drop(columns=[col for col in train.columns if col.startswith("Unnamed")], errors="ignore")
        test = test.drop(columns=[col for col in test.columns if col.startswith("Unnamed")], errors="ignore")

        # Validate data
        if train[date_col].isna().any() or test[date_col].isna().any():
            st.error(f"Invalid dates in '{date_col}' column.")
            return None, None, None, None, None, None
        if not pd.to_numeric(train[target_col], errors="coerce").notna().all():
            st.error(f"Target column '{target_col}' contains non-numeric values.")
            return None, None, None, None, None, None

        # Sample train data
        if len(train) > MAX_TRAIN_ROWS:
            train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

        # Feature engineering
        for df in [train, test]:
            df["month"] = df[date_col].dt.month.astype("int8")
            df["day"] = df[date_col].dt.day.astype("int8")
            df["dow"] = df[date_col].dt.dayofweek.astype("int8")
            df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")

        # Encode categoricals
        le_store = LabelEncoder()
        le_family = LabelEncoder()
        train["store_nbr_encoded"] = le_store.fit_transform(train["store_nbr"]).astype("int8")
        test["store_nbr_encoded"] = le_store.transform(test["store_nbr"]).astype("int8")
        train["family_encoded"] = le_family.fit_transform(train["family"]).astype("int8")
        test["family_encoded"] = le_family.transform(test["family"]).astype("int8")

        # Encode optional categoricals
        for col in ["city", "state", "locale"]:
            if col in train.columns and col in test.columns:
                le = LabelEncoder()
                train[f"{col}_encoded"] = le.fit_transform(train[col]).astype("int8")
                test[f"{col}_encoded"] = le.transform(test[col]).astype("int8")

        # Lag features
        train = train.sort_values([date_col])
        test = test.sort_values([date_col])
        for lag in [7, 14]:
            train[f"lag_{lag}"] = train.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0).astype("float32")
            test[f"lag_{lag}"] = 0.0 if target_col not in test else \
                test.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0).astype("float32")

        # Features for XGBoost/LSTM
        feature_cols = ["store_nbr_encoded", "family_encoded", "onpromotion", 
                        "month", "day", "dow", "is_weekend", "lag_7", "lag_14"]
        for col in ["city_encoded", "state_encoded", "locale_encoded"]:
            if col in train.columns:
                feature_cols.append(col)

        # Scale features
        scaler = StandardScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols]).astype("float32")
        test[feature_cols] = scaler.transform(test[feature_cols]).astype("float32")

        return train, test, feature_cols, scaler, le_store, le_family

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None, None, None

def prepare_lstm_data(data, target_col, feature_cols, sequence_length=14):
    """Prepare sequences for LSTM model."""
    X, y = [], []
    data_array = data[feature_cols + [target_col]].values
    for i in range(len(data) - sequence_length):
        X.append(data_array[i:i + sequence_length, :-1])
        y.append(data_array[i + sequence_length, -1])
    return np.array(X), np.array(y)

def train_and_predict(train, test, feature_cols, date_col, target_col, model_name):
    """Train specified model and generate predictions."""
    try:
        predictions = []
        test = test.sort_values(["store_nbr", "family", date_col])
        actuals = test[target_col].values if target_col in test.columns else None

        for (store, family), test_group in test.groupby(["store_nbr", "family"]):
            train_group = train[(train["store_nbr"] == store) & (train["family"] == family)]
            test_group = test_group.sort_values(date_col)

            if len(train_group) < MIN_SAMPLES or train_group[target_col].var() == 0:
                preds = np.zeros(len(test_group))
            else:
                if model_name == "XGBoost":
                    X_train = train_group[feature_cols]
                    y_train = np.log1p(train_group[target_col].clip(0))
                    X_test = test_group[feature_cols]
                    model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                    model.fit(X_train, y_train)
                    preds_log = model.predict(X_test)
                    preds = np.expm1(preds_log).clip(0)
                elif model_name == "ARIMA":
                    model = ARIMA(train_group[target_col], order=(3, 1, 0))
                    fit = model.fit()
                    preds = fit.forecast(steps=len(test_group)).clip(0)
                elif model_name == "SARIMA":
                    model = SARIMAX(train_group[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                    fit = model.fit(disp=False)
                    preds = fit.forecast(steps=len(test_group)).clip(0)
                elif model_name == "Prophet":
                    df = train_group[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
                    model = Prophet(daily_seasonality=True)
                    model.fit(df)
                    future = pd.DataFrame({"ds": test_group[date_col]})
                    forecast = model.predict(future)
                    preds = forecast["yhat"].values.clip(0)
                elif model_name == "LSTM":
                    sequence_length = 14
                    X_train, y_train = prepare_lstm_data(train_group, target_col, feature_cols, sequence_length)
                    if len(X_train) == 0:
                        preds = np.zeros(len(test_group))
                    else:
                        model = Sequential([
                            LSTM(50, activation="relu", input_shape=(sequence_length, len(feature_cols))),
                            Dense(1)
                        ])
                        model.compile(optimizer="adam", loss="mse")
                        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                        combined = pd.concat([train_group.tail(sequence_length), test_group]).reset_index(drop=True)
                        X_test, _ = prepare_lstm_data(combined, target_col, feature_cols + [target_col], sequence_length)
                        X_test = X_test[-len(test_group):]
                        preds = model.predict(X_test, verbose=0).flatten().clip(0)

            group_preds = pd.DataFrame({
                "id": test_group["id"],
                date_col: test_group[date_col],
                target_col: preds,
                "store_nbr": test_group["store_nbr"],
                "family": test_group["family"]
            })
            if actuals is not None:
                group_preds["actual"] = test_group[target_col] if target_col in test_group.columns else np.nan
            predictions.append(group_preds)

        predictions_df = pd.concat(predictions).sort_values("id")
        return predictions_df

    except Exception as e:
        st.error(f"Error during training/prediction with {model_name}: {str(e)}")
        return None

def to_csv_download(df):
    """Convert DataFrame to CSV for download."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

# Streamlit UI
training_tab, prediction_tab = st.tabs(["Training", "Prediction"])

with training_tab:
    st.header("Train Forecasting Models")

    # Upload Data Buttons
    train_file = st.file_uploader("Upload Train CSV", type="csv", key="uploader_train")
    test_file = st.file_uploader("Upload Test CSV", type="csv", key="uploader_test")

    # Column selection
    date_column = None
    target_column = None
    if train_file and test_file:
        try:
            train_df = pd.read_csv(train_file, nrows=1)
            columns = train_df.columns.tolist()
            st.subheader("Select Columns")
            date_column = st.selectbox("Select Date Column", columns, 
                                     index=columns.index("date") if "date" in columns else 0,
                                     key="date_column_select")
            target_column = st.selectbox("Select Target Column (Numeric)", columns,
                                       index=columns.index("sales") if "sales" in columns else 0,
                                       key="target_column_select")
            st.session_state.date_column = date_column
            st.session_state.target_column = target_column
        except Exception as e:
            st.error(f"Error reading train.csv: {str(e)}")
            st.stop()

    # Model selection
    selected_model = st.selectbox("Select Model to Train", MODELS, key="train_model_select")
    train_button = st.button("Generate Predictions")

    if train_button and train_file and test_file and selected_model and date_column and target_column:
        with st.spinner("Processing data..."):
            result = load_and_process_data(train_file, test_file, date_column, target_column)
            if result[0] is None:
                st.stop()
            train, test, feature_cols, scaler, le_store, le_family = result

            st.session_state.train = train
            st.session_state.test = test
            st.session_state.feature_cols = feature_cols
            st.session_state.scaler = scaler
            st.session_state.le_store = le_store
            st.session_state.le_family = le_family

            st.subheader("Processed Data Preview")
            st.dataframe(train.head(100))
            st.download_button(
                label="Download Processed Data as CSV",
                data=to_csv_download(train),
                file_name="processed_train.csv",
                mime="text/csv"
            )

            with st.spinner(f"Training {selected_model}..."):
                predictions = train_and_predict(train, test, feature_cols, date_column, target_column, selected_model)
                if predictions is None:
                    st.stop()
                st.session_state.predictions[selected_model] = predictions

                # Plot predictions (first 100 points)
                fig = go.Figure()
                if "actual" in predictions.columns:
                    fig.add_trace(go.Scatter(x=predictions[date_column][:100], y=predictions["actual"][:100], 
                                           mode="lines", name="Actual", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=predictions[date_column][:100], y=predictions[target_column][:100], 
                                       mode="lines", name="Predicted", line=dict(color="orange")))
                fig.update_layout(
                    title=f"{selected_model} Predictions",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    xaxis_tickangle=45,
                    yaxis_gridcolor="lightgray"
                )
                st.plotly_chart(fig)

                # Calculate and display metrics
                if "actual" in predictions.columns:
                    actuals = predictions["actual"].values
                    preds = predictions[target_column].values
                    rmsle = np.sqrt(mean_squared_error(np.log1p(actuals), np.log1p(preds)))
                    rmse = np.sqrt(mean_squared_error(actuals, preds))
                    mae = mean_absolute_error(actuals, preds)
                    st.write(f"### {selected_model} Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSLE", f"{rmsle:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")

                # Download predictions
                st.download_button(
                    label="Download Predictions as CSV",
                    data=to_csv_download(predictions),
                    file_name=f"predictions_{selected_model}.csv",
                    mime="text/csv"
                )

            st.success("Training completed!")

with prediction_tab:
    st.header("Visualize Predictions")

    if st.session_state.train is not None:
        store_nbrs = sorted(st.session_state.train["store_nbr"].unique())
        families = sorted(st.session_state.train["family"].unique())

        st.subheader("Select Parameters")
        store_nbr = st.selectbox("Store Number", store_nbrs, key="viz_store")
        family = st.selectbox("Product Family", families, key="viz_family")
        selected_model = st.selectbox("Model", MODELS, key="viz_model")

        if selected_model in st.session_state.predictions:
            predictions = st.session_state.predictions[selected_model]
            subset = predictions[(predictions["store_nbr"] == store_nbr) & (predictions["family"] == family)]

            if not subset.empty:
                st.subheader(f"{selected_model} Predictions for Store {store_nbr}, Family {family}")
                fig = go.Figure()
                if "actual" in subset.columns:
                    fig.add_trace(go.Scatter(x=subset[st.session_state.date_column][:100], 
                                           y=subset["actual"][:100], 
                                           mode="lines", name="Actual", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=subset[st.session_state.date_column][:100], 
                                       y=subset[st.session_state.target_column][:100], 
                                       mode="lines", name="Predicted", line=dict(color="orange")))
                fig.update_layout(
                    title=f"Predictions: Store {store_nbr}, Family {family}",
                    xaxis_title="Date",
                    yaxis_title=st.session_state.target_column,
                    xaxis_tickangle=45,
                    yaxis_gridcolor="lightgray"
                )
                st.plotly_chart(fig)

                # Display metrics for subset if actuals are available
                if "actual" in subset.columns:
                    actuals = subset["actual"].values
                    preds = subset[st.session_state.target_column].values
                    rmsle = np.sqrt(mean_squared_error(np.log1p(actuals), np.log1p(preds)))
                    rmse = np.sqrt(mean_squared_error(actuals, preds))
                    mae = mean_absolute_error(actuals, preds)
                    st.write("### Metrics for Selected Group")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSLE", f"{rmsle:.4f}")
                    col2.metric("RMSE", f"{rmse:.4f}")
                    col3.metric("MAE", f"{mae:.4f}")
            else:
                st.warning("No predictions available for this store-family pair.")
        else:
            st.warning(f"No predictions available for {selected_model}. Please train the model first.")
