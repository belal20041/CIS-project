import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Constants
MAX_TRAIN_ROWS = 150000
MIN_SAMPLES = 14

def load_and_process_data(train_path, test_path, date_col="date", target_col="sales"):
    """Load and process train/test CSV files."""
    try:
        # Define dtypes for efficiency
        dtypes = {
            "store_nbr": "int32",
            "family": "category",
            "onpromotion": "int32",
            target_col: "float32",
            "id": "int32"
        }
        optional_cols = ["city", "state", "type_x", "cluster", "transactions", "dcoilwtico", 
                         "locale", "locale_name", "description", "transferred", "type_y"]
        for col in optional_cols:
            dtypes[col] = ("category" if col in ["city", "state", "type_x", "locale", 
                                                "locale_name", "description", "type_y"]
                          else "float32" if col in ["dcoilwtico", "transactions", "cluster"]
                          else "bool")

        # Load data
        train = pd.read_csv(train_path, dtype=dtypes, parse_dates=[date_col])
        test = pd.read_csv(test_path, dtype=dtypes, parse_dates=[date_col])

        # Drop Unnamed: 17 if present
        if "Unnamed: 17" in train.columns:
            train = train.drop(columns=["Unnamed: 17"])
        if "Unnamed: 17" in test.columns:
            test = test.drop(columns=["Unnamed: 17"])

        # Validate target column
        if not pd.to_numeric(train[target_col], errors="coerce").notna().all():
            raise ValueError(f"Target column '{target_col}' contains non-numeric values.")

        # Sample train data to reduce memory usage
        if len(train) > MAX_TRAIN_ROWS:
            train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

        # Basic feature engineering
        for df in [train, test]:
            df["month"] = df[date_col].dt.month.astype("int8")
            df["day"] = df[date_col].dt.day.astype("int8")
            df["dow"] = df[date_col].dt.dayofweek.astype("int8")
            df["is_weekend"] = df[date_col].dt.dayofweek.isin([5, 6]).astype("int8")

        # Encode categorical columns
        le_store = LabelEncoder()
        le_family = LabelEncoder()
        train["store_nbr_encoded"] = le_store.fit_transform(train["store_nbr"]).astype("int8")
        test["store_nbr_encoded"] = le_store.transform(test["store_nbr"]).astype("int8")
        train["family_encoded"] = le_family.fit_transform(train["family"]).astype("int8")
        test["family_encoded"] = le_family.transform(test["family"]).astype("int8")

        # Simple lag features
        train = train.sort_values([date_col])
        test = test.sort_values([date_col])
        for lag in [7, 14]:
            train[f"lag_{lag}"] = train.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0).astype("float32")
            test[f"lag_{lag}"] = test.groupby(["store_nbr", "family"])[target_col].shift(lag).fillna(0).astype("float32") if target_col in test else 0

        # Define feature columns
        feature_cols = ["store_nbr_encoded", "family_encoded", "onpromotion", "month", "day", "dow", 
                        "is_weekend", "lag_7", "lag_14"]

        # Scale features
        scaler = StandardScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols]).astype("float32")
        test[feature_cols] = scaler.transform(test[feature_cols]).astype("float32")

        return train, test, feature_cols, scaler, le_store, le_family

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None, None, None, None, None, None

def train_and_predict(train, test, feature_cols, date_col, target_col, min_samples):
    """Train XGBoost model and generate predictions."""
    try:
        predictions = []
        test_ids = test["id"].values
        test_dates = test[date_col].values

        # Group by store and family
        for (store, family), group in test.groupby(["store_nbr", "family"]):
            train_group = train[(train["store_nbr"] == store) & (train["family"] == family)]
            test_group = group.sort_values(date_col)

            if len(train_group) >= min_samples and train_group[target_col].var() > 0:
                # Prepare data
                X_train = train_group[feature_cols]
                y_train = np.log1p(train_group[target_col].clip(0))
                X_test = test_group[feature_cols]

                # Train XGBoost
                model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)

                # Predict
                preds_log = model.predict(X_test)
                preds = np.expm1(preds_log).clip(0)
            else:
                preds = np.zeros(len(test_group))

            # Store predictions
            group_preds = pd.DataFrame({
                "id": test_group["id"],
                "date": test_group[date_col],
                target_col: preds
            })
            predictions.append(group_preds)

        # Combine predictions
        predictions_df = pd.concat(predictions).sort_values("id")
        return predictions_df

    except Exception as e:
        print(f"Error during training/prediction: {str(e)}")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sales Forecasting Script")
    parser.add_argument("--train", required=True, help="Path to train CSV file")
    parser.add_argument("--test", required=True, help="Path to test CSV file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file for predictions")
    parser.add_argument("--date-col", default="date", help="Name of date column")
    parser.add_argument("--target-col", default="sales", help="Name of target column")
    args = parser.parse_args()

    # Load and process data
    print("Loading and processing data...")
    train, test, feature_cols, scaler, le_store, le_family = load_and_process_data(
        args.train, args.test, args.date_col, args.target_col
    )
    if train is None:
        print("Failed to process data. Exiting.")
        return

    # Train and predict
    print("Training model and generating predictions...")
    predictions = train_and_predict(train, test, feature_cols, args.date_col, args.target_col, MIN_SAMPLES)
    if predictions is None:
        print("Failed to generate predictions. Exiting.")
        return

    # Save predictions
    predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Calculate metrics (if target is available in test for validation)
    if args.target_col in test.columns:
        actuals = test[args.target_col].values
        preds = predictions[args.target_col].values
        rmsle = np.sqrt(mean_squared_error(np.log1p(actuals), np.log1p(preds)))
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        print(f"Metrics: RMSLE={rmsle:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

if __name__ == "__main__":
    main()
