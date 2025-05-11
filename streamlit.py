import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, chi2_contingency, ks_2samp
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing xgboost, handle if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Install it with `pip install xgboost` or `conda install -c conda-forge xgboost`. Only Linear Regression and Random Forest models will be available.")

# Cache data loading with cleaning
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Drop empty 'Unnamed: *' columns
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:') and df[col].isna().all()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        st.info(f"Dropped empty columns: {unnamed_cols}")
    return df

# Validate numeric target
def validate_target(df, target_col):
    return df[target_col].dtype in [np.float64, np.float32, np.int64, np.int32]

# Preprocessing pipeline
def create_preprocessing_pipeline(numeric_cols, categorical_cols, fill_strategy):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=fill_strategy)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

# Preprocess data
def preprocess_data(df, drop_cols, fill_method, date_col, target_col):
    df = df.drop(columns=drop_cols, errors='ignore')
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # Exclude target column from numeric columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    if fill_method != 'none' and len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy=fill_method)
        try:
            imputed = imputer.fit_transform(df[numeric_cols])
            if len(numeric_cols) == 1:
                df[numeric_cols[0]] = imputed.ravel()
            else:
                df[numeric_cols] = imputed
        except Exception as e:
            st.error(f"Error during imputation: {e}")
            return df
    df = df.drop_duplicates()
    st.write(f"Processed columns: {list(df.columns)}")
    return df

# Statistical tests for regression
def run_statistical_tests(df, target_col, numeric_cols, categorical_cols):
    results = []
    for col in numeric_cols:
        if col != target_col and col in df.columns:
            valid_data = df[[col, target_col]].dropna()
            if len(valid_data) > 1:
                corr, p_val = pearsonr(valid_data[col], valid_data[target_col])
                results.append({'Feature': col, 'Test': 'Pearson Correlation', 'P-value': p_val, 'Correlation': corr})
    for col in categorical_cols:
        if col in df.columns:
            try:
                df['target_binned'] = pd.qcut(df[target_col], q=4, duplicates='drop')
                contingency_table = pd.crosstab(df[col], df['target_binned'])
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                results.append({'Feature': col, 'Test': 'Chi-squared', 'P-value': p_val})
            except:
                results.append({'Feature': col, 'Test': 'Chi-squared', 'P-value': np.nan})
            df = df.drop(columns=['target_binned'], errors='ignore')
    return pd.DataFrame(results)

# Updated feature engineering
def engineer_features(df, date_col, numeric_cols, categorical_cols, target_col):
    df = df.copy()
    # Validate columns
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns and col != target_col]
    valid_categorical_cols = [col for col in categorical_cols if col in df.columns]
    if not valid_numeric_cols and not valid_categorical_cols:
        st.warning("No valid features for engineering. Skipping feature creation.")
        return df
    
    if date_col and date_col in df.columns:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['tenure'] = (pd.to_datetime('today') - df[date_col]).dt.days / 30.0
    for col in valid_numeric_cols:
        df[f'{col}_squared'] = df[col] ** 2
    for i, col1 in enumerate(valid_numeric_cols):
        for col2 in valid_numeric_cols[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    if date_col and date_col in df.columns:
        df = df.sort_values(by=date_col)
        df[f'{target_col}_lag1'] = df[target_col].shift(1)
    return df

# Model training and evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test, model_choice, preprocessor):
    model_dict = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor()
    }
    if XGBOOST_AVAILABLE:
        model_dict['XGBoost'] = xgb.XGBRegressor(eval_metric='rmse')
    
    param_grids = {
        'Linear Regression': {'model__fit_intercept': [True, False]},
        'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [5, 10]}
    }
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_dict[model_choice])
    ])
    grid_search = GridSearchCV(pipeline, param_grids[model_choice], cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return grid_search.best_estimator_, metrics, y_pred

# Simulate monitoring
def monitor_performance(original_df, new_df, target_col, model, preprocessor):
    if new_df is not None and target_col in new_df:
        X_new = new_df.drop(columns=[target_col])
        y_new = new_df[target_col]
        try:
            X_new_processed = preprocessor.transform(X_new)
            y_pred_new = model.predict(X_new_processed)
            r2_new = r2_score(y_new, y_pred_new)
            
            drift_results = []
            for col in original_df.columns:
                if col in new_df and col != target_col:
                    stat, p_val = ks_2samp(original_df[col].dropna(), new_df[col].dropna())
                    drift_results.append({'Feature': col, 'KS P-value': p_val})
            return r2_new, pd.DataFrame(drift_results)
        except Exception as e:
            st.error(f"Monitoring error: {e}")
            return None, None
    return None, None

# Main app
def main():
    st.title("Sales Prediction and Analysis App")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
    if uploaded_file is None:
        st.info("Please upload a CSV file with a numeric 'sales' column.")
        return
    
    df = load_data(uploaded_file)
    st.success("Dataset Loaded Successfully!")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Target and feature selection
    target_col = st.sidebar.selectbox("Select Target Column (Sales)", df.columns, index=df.columns.get_loc('sales') if 'sales' in df.columns else 0)
    if not validate_target(df, target_col):
        st.error("Target column 'sales' must be numeric (float or integer).")
        return
    
    # Preprocessing (to get valid columns)
    st.sidebar.subheader("Preprocessing")
    drop_cols = st.sidebar.multiselect("Columns to Drop", df.columns)
    fill_strategy = st.sidebar.selectbox("Fill Missing Values", ["mean", "median", "most_frequent"])
    date_col = st.sidebar.selectbox("Date Column (optional)", [None] + df.columns.tolist())
    
    df_processed = preprocess_data(df, drop_cols, fill_strategy, date_col, target_col)
    
    # Feature selection based on processed data
    default_numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns if col != target_col]
    default_categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = st.sidebar.multiselect("Numeric Features", default_numeric_cols, default=default_numeric_cols)
    categorical_cols = st.sidebar.multiselect("Categorical Features", default_categorical_cols, default=default_categorical_cols)
    
    # Outlier handling
    st.sidebar.subheader("Outlier Handling")
    outlier_cols = st.sidebar.multiselect("Outlier Columns", numeric_cols)
    outlier_method = st.sidebar.selectbox("Outlier Method", ["none", "remove_iqr", "replace_median"])
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = ["Linear Regression", "Random Forest"]
    if XGBOOST_AVAILABLE:
        model_options.append("XGBoost")
    model_choices = st.sidebar.multiselect("Models", model_options, default=["Linear Regression"])
    
    # Validation data for monitoring
    st.sidebar.subheader("Monitoring")
    validation_file = st.sidebar.file_uploader("Upload Validation CSV (optional)", type="csv")
    validation_df = load_data(validation_file) if validation_file else None
    
    # Tabs
    tabs = st.tabs(["Home", "EDA", "Preprocessing", "Analysis", "Modeling", "Deployment", "Report"])
    
    # Home
    with tabs[0]:
        st.write("### Welcome")
        st.write("This app predicts sales using machine learning. Upload a dataset with a numeric 'sales' column, configure settings, and explore results across tabs.")
    
    # EDA
    with tabs[1]:
        st.write("### Exploratory Data Analysis")
        st.write("**Dataset Summary**")
        st.write(f"Shape: {df.shape}")
        st.write("Missing Values:", df.isna().sum())
        st.write("Duplicates:", df.duplicated().sum())
        
        st.write("**Sales Distribution**")
        fig = px.histogram(df, x=target_col, title="Sales Distribution")
        st.plotly_chart(fig)
        
        if numeric_cols:
            st.write("**Numeric Features vs. Sales**")
            for col in numeric_cols[:3]:
                if col in df.columns:
                    fig = px.scatter(df, x=col, y=target_col, title=f"{col} vs. Sales")
                    st.plotly_chart(fig)
        
        if categorical_cols:
            st.write("**Categorical Features vs. Average Sales**")
            for col in categorical_cols[:3]:
                if col in df.columns:
                    avg_sales = df.groupby(col)[target_col].mean().reset_index()
                    fig = px.bar(avg_sales, x=col, y=target_col, title=f"Average Sales by {col}")
                    st.plotly_chart(fig)
        
        if numeric_cols:
            st.write("**Correlation Heatmap**")
            corr_cols = [col for col in numeric_cols if col in df.columns] + [target_col]
            if len(corr_cols) > 1:
                corr = df[corr_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig)
    
    # Preprocessing
    with tabs[2]:
        st.write("### Preprocessing")
        if outlier_cols and outlier_method != "none":
            for col in outlier_cols:
                if col in df_processed.columns:
                    if outlier_method == "remove_iqr":
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df_processed = df_processed[(df_processed[col] >= Q1 - 1.5 * IQR) & (df_processed[col] <= Q3 + 1.5 * IQR)]
                    elif outlier_method == "replace_median":
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        median = df_processed[col].median()
                        df_processed[col] = df_processed[col].apply(lambda x: median if x < Q1 - 1.5 * IQR or x > Q3 + 1.5 * IQR else x)
        
        preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols, fill_strategy)
        st.write("**Processed Data Preview**")
        st.dataframe(df_processed.head())
    
    # Advanced Analysis and Feature Engineering
    with tabs[3]:
        st.write("### Advanced Analysis and Feature Engineering")
        df_engineered = engineer_features(df_processed, date_col, numeric_cols, categorical_cols, target_col)
        
        st.write("**Statistical Tests**")
        stat_results = run_statistical_tests(df_processed, target_col, numeric_cols, categorical_cols)
        st.dataframe(stat_results)
        
        st.write("**Feature Importance (RFE)**")
        X = df_engineered.drop(columns=[target_col])
        y = df_engineered[target_col]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        total_features = len(numeric_cols) + len(categorical_cols)
        if total_features >= 2:
            try:
                rfe = RFE(estimator=pipeline.steps[1][1], n_features_to_select=min(5, total_features))
                X_processed = preprocessor.fit_transform(X)
                rfe.fit(X_processed, y)
                feature_ranking = pd.DataFrame({
                    'Feature': preprocessor.get_feature_names_out(),
                    'Ranking': rfe.ranking_
                })
                st.dataframe(feature_ranking)
            except Exception as e:
                st.error(f"RFE error: {e}")
        else:
            st.warning("At least two features are required for RFE. Please select more numeric or categorical features.")
        
        st.write("**New Features**")
        new_cols = [col for col in df_engineered.columns if col not in df.columns]
        st.write(new_cols)
        st.dataframe(df_engineered[new_cols].head())
    
    # Modeling
    with tabs[4]:
        st.write("### Model Development and Evaluation")
        X = df_engineered.drop(columns=[target_col])
        y = df_engineered[target_col]
        if total_features == 0:
            st.error("No features selected. Please select at least one numeric or categorical feature.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        for model_choice in model_choices:
            model, metrics, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice, preprocessor)
            results.append({'Model': model_choice, **metrics})
            
            st.write(f"**{model_choice} Results**")
            st.write(metrics)
            
            st.write("**Predicted vs. Actual Sales**")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, title=f"{model_choice} Predicted vs. Actual")
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal'))
            st.plotly_chart(fig)
            
            st.write("**Residual Plot**")
            residuals = y_test - y_pred
            fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Sales', 'y': 'Residuals'}, title=f"{model_choice} Residuals")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig)
            
            joblib.dump(model, f"{model_choice}_model.pkl")
        
        st.write("**Model Comparison**")
        st.dataframe(pd.DataFrame(results))
    
    # Deployment and Monitoring
    with tabs[5]:
        st.write("### Deployment and Monitoring")
        selected_model = st.selectbox("Select Model for Prediction", model_choices)
        model = joblib.load(f"{selected_model}_model.pkl")
        
        st.write("**Live Prediction**")
        input_data = {}
        for col in X.columns:
            if col in numeric_cols:
                input_data[col] = st.number_input(f"Input {col}", value=0.0)
            else:
                input_data[col] = st.text_input(f"Input {col}", value="Unknown")
        
        if st.button("Predict Sales"):
            input_df = pd.DataFrame([input_data])
            try:
                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)[0]
                st.write(f"Predicted Sales: {prediction:.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        
        st.write("**Monitoring**")
        r2_new, drift_results = monitor_performance(df, validation_df, target_col, model, preprocessor)
        if r2_new is not None:
            st.write(f"Validation R²: {r2_new:.2f}")
            if r2_new < 0.7:
                st.warning("Model performance degraded. Consider retraining.")
            st.write("**Data Drift Analysis**")
            st.dataframe(drift_results)
    
    # Report
    with tabs[6]:
        st.write("### Final Report")
        report = {
            "Dataset": f"Shape: {df.shape}, Columns: {list(df.columns)}",
            "EDA Insights": f"Missing Values: {df.isna().sum().sum()}, Duplicates: {df.duplicated().sum()}",
            "Statistical Analysis": f"Significant Features: {stat_results[stat_results['P-value'] < 0.05]['Feature'].tolist()}",
            "Model Performance": results,
            "Business Impact": "Predicting sales enables better inventory management, demand forecasting, and revenue optimization.",
            "Future Improvements": ["Incorporate time-series models", "Add external economic indicators", "Deploy on cloud platforms"]
        }
        st.json(report)
        st.download_button("Download Report", json.dumps(report, indent=2), "sales_report.json", "application/json")

if __name__ == "__main__":
    main()