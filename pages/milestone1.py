# pages/milestone1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
import os

st.set_page_config(page_title="Milestone 1: Data Collection, Exploration, Preprocessing", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>MILESTONE 1: Data Collection, Exploration, Preprocessing</h1>", unsafe_allow_html=True)

def load_data(file):
    """Load dataset from uploaded file."""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.parquet'):
        return pd.read_parquet(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Parquet file.")
        return None

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None):
    """Perform exploratory data analysis."""
    st.subheader("Data Exploration")
    st.write("**Shape**:", df.shape)
    st.write("**Info**:")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Type', 'index': 'Column'}))
    st.write("**Summary Statistics**:")
    st.dataframe(df.describe())

    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            st.write("**Date Column Distribution**:")
            fig, ax = plt.subplots()
            df[date_col].dropna().hist(ax=ax)
            st.pyplot(fig)
        except:
            st.warning(f"Could not convert {date_col} to datetime.")

    st.write("**Missing Values**:")
    st.write(df.isna().sum())
    fig, ax = plt.subplots()
    msno.matrix(df, ax=ax)
    st.pyplot(fig)

    st.write("**Duplicates**:")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

    if numeric_cols:
        st.write("**Boxplots for Numeric Columns**:")
        fig, axes = plt.subplots(2, (len(numeric_cols) + 1) // 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        st.pyplot(fig)

        st.write("**Outlier Detection**:")
        outliers_iqr = {}
        outliers_zscore = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if df[col].std() > 1e-6:
                z_scores = zscore(df[col])
                outliers_zscore[col] = df[abs(z_scores) > 3][col]
            else:
                outliers_zscore[col] = pd.Series(dtype=df[col].dtype)
            st.write(f"{col}: IQR Outliers = {len(outliers_iqr[col])}, Z-score Outliers = {len(outliers_zscore[col])}")

    if categorical_cols:
        st.write("**Categorical Column Distributions**:")
        for col in categorical_cols:
            st.write(f"**{col} Value Counts**:")
            st.dataframe(df[col].value_counts())

def preprocess_data(df, numeric_cols=None, categorical_cols=None, date_col=None, handle_outliers='remove'):
    """Preprocess the dataset."""
    df_clean = df.copy()

    # Handle missing values
    st.subheader("Preprocessing: Missing Values")
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    st.write("Missing values filled.")

    # Remove duplicates
    st.subheader("Preprocessing: Duplicates")
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    st.write(f"Removed {initial_rows - df_clean.shape[0]} duplicate rows.")

    # Handle outliers
    if numeric_cols and handle_outliers:
        st.subheader("Preprocessing: Outliers")
        if handle_outliers == 'remove':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            st.write(f"After removing outliers: {df_clean.shape[0]} rows")
        elif handle_outliers == 'replace':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].apply(
                    lambda x: median_value if (x < lower_bound or x > upper_bound) else x
                )
            st.write(f"After replacing outliers with median: {df_clean.shape[0]} rows")

    # Add time features if date column is provided
    if date_col:
        try:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean['day'] = df_clean[date_col].dt.day
            df_clean['month'] = df_clean[date_col].dt.month
            df_clean['year'] = df_clean[date_col].dt.year
            df_clean['dayofweek'] = df_clean[date_col].dt.dayofweek
            df_clean['sin_month'] = np.sin(2 * np.pi * df_clean['month'] / 12)
            st.write("Added time-based features: day, month, year, dayofweek, sin_month")
        except:
            st.warning(f"Could not process time features for {date_col}.")

    return df_clean

def main():
    st.markdown("### Objectives")
    st.write("- Collect, explore, and preprocess historical sales data for analysis and modeling.")

    # Data Collection
    st.markdown("### 1. Data Collection")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Parquet)", type=['csv', 'parquet'])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df
            st.write("Dataset loaded successfully!")
            st.dataframe(df.head())

    # Column Selection
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.markdown("### Column Selection")
        date_col = st.selectbox("Select Date Column (optional)", ['None'] + list(df.columns), index=0)
        numeric_cols = st.multiselect("Select Numeric Columns", df.columns)
        categorical_cols = st.multiselect("Select Categorical Columns", df.columns)
        date_col = None if date_col == 'None' else date_col

        # Data Exploration
        if st.button("Run Data Exploration"):
            explore_data(df, date_col, numeric_cols, categorical_cols)

        # Data Preprocessing
        st.markdown("### 3. Data Preprocessing")
        handle_outliers = st.selectbox("Handle Outliers", ['None', 'Remove', 'Replace with Median'])
        handle_outliers = handle_outliers.lower() if handle_outliers != 'None' else None
        if st.button("Preprocess Data"):
            processed_df = preprocess_data(df, numeric_cols, categorical_cols, date_col, handle_outliers)
            st.session_state['processed_df'] = processed_df
            st.write("Preprocessing complete!")
            st.dataframe(processed_df.head())

            # Save processed data
            save_option = st.selectbox("Save Processed Data As", ['CSV', 'Parquet'])
            if st.button("Save Processed Data"):
                output_path = os.path.join("data", f"processed_{uploaded_file.name.split('.')[0]}.{save_option.lower()}")
                if save_option == 'CSV':
                    processed_df.to_csv(output_path, index=False)
                else:
                    processed_df.to_parquet(output_path, index=False)
                st.success(f"Processed data saved to {output_path}")
                # Add to DVC
                os.system(f"dvc add {output_path}")
                os.system(f"git add {output_path}.dvc")
                os.system('git commit -m "Add processed dataset to DVC"')

if __name__ == "__main__":
    main() 
