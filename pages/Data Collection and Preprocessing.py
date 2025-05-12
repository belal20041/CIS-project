# pages/milestone1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Milestone 1: Data Collection, Exploration, Preprocessing", layout="wide")
st.markdown("<div style='text-align: center; font-size: 28px; color: #2E86C1;'>MILESTONE 1: Data Collection, Exploration, Preprocessing</div>", unsafe_allow_html=True)

def load_data(file, dataset_type="train"):
    """Load dataset from uploaded file. Assigned to: Mohamed Samy (Standardization and Formatting)"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Parquet file.")
        return None
    # Fix inconsistencies (e.g., drop unnamed columns). Assigned to: Hoda Magdy
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def detect_column_types(df, date_col=None):
    """Automatically detect numeric and categorical columns. Assigned to: Mohamed Samy"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    # Identify categorical columns: object, category, or columns with few unique values
    categorical_cols = []
    for col in df.columns:
        if col == date_col:
            continue
        if df[col].dtype in ['object', 'category', 'bool']:
            categorical_cols.append(col)
        elif df[col].nunique() / len(df[col]) < 0.05:  # Less than 5% unique values
            categorical_cols.append(col)
    
    # Remove date column from numeric/categorical if present
    if date_col:
        if date_col in numeric_cols:
            numeric_cols.remove(date_col)
        if date_col in categorical_cols:
            categorical_cols.remove(date_col)
    
    return numeric_cols, categorical_cols

def apply_column_type_changes(df, type_changes, date_col=None):
    """Apply user-specified column type changes. Assigned to: Mohamed Samy"""
    df_modified = df.copy()
    new_numeric_cols = []
    new_categorical_cols = []
    
    for col, new_type in type_changes.items():
        if col == date_col and new_type != 'date':
            st.warning(f"Cannot change type of date column {col} to {new_type}. Keeping as date.")
            continue
        try:
            if new_type == 'numeric':
                df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce')
                new_numeric_cols.append(col)
            elif new_type == 'categorical':
                df_modified[col] = df_modified[col].astype('category')
                new_categorical_cols.append(col)
            elif new_type == 'date' and col != date_col:
                df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')
                if date_col is None:
                    date_col = col
        except Exception as e:
            st.error(f"Failed to convert {col} to {new_type}: {str(e)}")
    
    numeric_cols, categorical_cols = detect_column_types(df_modified, date_col)
    numeric_cols = list(set(numeric_cols + new_numeric_cols))
    categorical_cols = list(set(categorical_cols + new_categorical_cols))
    
    if date_col:
        if date_col in numeric_cols:
            numeric_cols.remove(date_col)
        if date_col in categorical_cols:
            categorical_cols.remove(date_col)
    
    return df_modified, numeric_cols, categorical_cols, date_col

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None, dataset_type="train"):
    """Perform EDA: trends, seasonality, missing values, outliers. Assigned to: Belal Khamis"""
    st.subheader(f"Data Exploration - {dataset_type.capitalize()} Data")
    st.write("**Shape**:", df.shape)
    st.write("**Info**:")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Type', 'index': 'Column'}))
    st.write("**Summary Statistics**:")
    st.dataframe(df.describe())

    # Missing values analysis. Assigned to: Belal Khamis
    st.write("**Missing Values**:")
    st.write(df.isna().sum())
    fig, ax = plt.subplots()
    msno.matrix(df, ax=ax)
    st.pyplot(fig)

    # Duplicates. Assigned to: Marwa Kotb
    st.write("**Duplicates**:")
    st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Date-based analysis (trends, seasonality)
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            st.write("**Sales Trends Over Time**:")
            if 'sales' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                df.groupby(date_col)['sales'].sum().plot(ax=ax)
                ax.set_title("Total Sales Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales")
                plt.tight_layout()
                st.pyplot(fig)

            st.write("**Seasonality (Monthly Sales)**:")
            df['month'] = df[date_col].dt.month
            if 'sales' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='month', y='sales', data=df, ax=ax)
                ax.set_title("Monthly Sales Distribution")
                plt.tight_layout()
                st.pyplot(fig)
        except:
            st.warning(f"Could not process {date_col} for trends/seasonality.")

    # Patterns: Sales vs. Promotions
    if 'sales' in df.columns and 'onpromotion' in df.columns:
        st.write("**Sales vs. Promotions**:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='onpromotion', y='sales', data=df, ax=ax)
        ax.set_title("Sales vs. Promotions")
        plt.tight_layout()
        st.pyplot(fig)

    # Holiday analysis
    if 'transferred' in df.columns:
        st.write("**Holiday (Transferred) Distribution**:")
        st.dataframe(df['transferred'].value_counts())

    # Outlier detection. Assigned to: Mahmoud Sabry
    if numeric_cols:
        valid_numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])]
        if not valid_numeric_cols:
            st.warning("No valid numeric columns for analysis.")
            return

        st.write("**Boxplots for Numeric Columns**:")
        fig, axes = plt.subplots(2, (len(valid_numeric_cols) + 1) // 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, col in enumerate(valid_numeric_cols):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        st.pyplot(fig)

        st.write("**Outlier Detection**:")
        outliers_iqr = {}
        outliers_zscore = {}
        for col in valid_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if df[col].std() > 1e-6:
                z_scores = zscore(df[col].dropna())
                outliers_zscore[col] = df[abs(z_scores) > 3][col]
            else:
                outliers_zscore[col] = pd.Series(dtype=df[col].dtype)
            st.write(f"{col}: IQR Outliers = {len(outliers_iqr[col])}, Z-score Outliers = {len(outliers_zscore[col])}")

    if categorical_cols:
        st.write("**Categorical Column Distributions**:")
        for col in categorical_cols:
            st.write(f"**{col} Value Counts**:")
            st.dataframe(df[col].value_counts())

def preprocess_data(df, numeric_cols=None, categorical_cols=None, date_col=None, handle_outliers='remove', normalize=False, dataset_type="train"):
    """Preprocess dataset: missing values, duplicates, time features, normalization. Assigned to: Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy"""
    df_clean = df.copy()

    # Handle missing values. Assigned to: Belal Khamis
    st.subheader(f"Preprocessing: Missing Values - {dataset_type.capitalize()}")
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
    st.write("Missing values filled.")

    # Remove duplicates. Assigned to: Marwa Kotb
    st.subheader(f"Preprocessing: Duplicates - {dataset_type.capitalize()}")
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    st.write(f"Removed {initial_rows - df_clean.shape[0]} duplicate rows.")

    # Handle outliers. Assigned to: Mahmoud Sabry
    if numeric_cols and handle_outliers:
        st.subheader(f"Preprocessing: Outliers - {dataset_type.capitalize()}")
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

    # Add time features. Assigned to: Mohamed Samy
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

    # Normalize numeric columns. Assigned to: Mohamed Samy
    if normalize and numeric_cols:
        st.subheader(f"Preprocessing: Normalization - {dataset_type.capitalize()}")
        scaler = MinMaxScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
        st.write("Numeric columns normalized using Min-Max scaling.")

    return df_clean

def main():
    st.markdown("### Objectives")
    st.write("- Collect, explore, and preprocess historical sales data for analysis and modeling.")
    
    st.markdown("### Team Assignments")
    st.write("- Notebook Outlines & Handling Missing Values: Belal Khamis")
    st.write("- Removing Duplicates: Marwa Kotb")
    st.write("- Handling Outliers: Mahmoud Sabry")
    st.write("- Standardization and Formatting: Mohamed Samy")
    st.write("- Fixing Typos and Inconsistencies: Hoda Magdy")

    # Data Collection
    st.markdown("<div style='text-align:center;border-radius:0px;padding:18px;background-color:#000000;color:#FFFFFF;font-family:ARIAL BOLD'>PART 01<br><br>Train Data</div>", unsafe_allow_html=True)
    train_file = st.file_uploader("Upload Train Dataset (CSV or Parquet)", type=['csv', 'parquet'], key="train")
    if train_file:
        train_df = load_data(train_file, "train")
        if train_df is not None:
            st.session_state['train_df'] = train_df
            st.write("Train dataset loaded successfully!")
            st.dataframe(train_df.head())

    st.markdown("<div style='text-align:center;border-radius:0px;padding:18px;background-color:#000000;color:#FFFFFF;font-family:ARIAL BOLD'>PART 02<br><br>Test Data</div>", unsafe_allow_html=True)
    test_file = st.file_uploader("Upload Test Dataset (CSV or Parquet)", type=['csv', 'parquet'], key="test")
    if test_file:
        test_df = load_data(test_file, "test")
        if test_df is not None:
            st.session_state['test_df'] = test_df
            st.write("Test dataset loaded successfully!")
            st.dataframe(test_df.head())

    # Column Selection and Processing
    if 'train_df' in st.session_state or 'test_df' in st.session_state:
        st.markdown("### Column Selection and Type Adjustment")
        
        # Train Data Processing
        if 'train_df' in st.session_state:
            train_df = st.session_state['train_df']
            st.subheader("Train Data Configuration")
            train_date_col = st.selectbox("Select Date Column for Train (optional)", ['None'] + list(train_df.columns), index=0, key="train_date")
            train_date_col = None if train_date_col == 'None' else train_date_col

            # Detect column types
            train_numeric_cols, train_categorical_cols = detect_column_types(train_df, train_date_col)
            st.write("**Detected Numeric Columns (Train)**:", train_numeric_cols)
            st.write("**Detected Categorical Columns (Train)**:", train_categorical_cols)

            # Allow column type changes
            st.markdown("#### Change Train Column Types (Optional)")
            train_type_changes = {}
            for col in train_df.columns:
                if col == train_date_col:
                    continue
                current_type = 'numeric' if col in train_numeric_cols else 'categorical' if col in train_categorical_cols else 'other'
                new_type = st.selectbox(
                    f"Type for {col} (Train, current: {current_type})",
                    ['Keep as is', 'Numeric', 'Categorical', 'Date'],
                    key=f"train_type_{col}"
                )
                if new_type != 'Keep as is':
                    train_type_changes[col] = new_type.lower()

            if st.button("Apply Train Column Type Changes"):
                train_df, train_numeric_cols, train_categorical_cols, train_date_col = apply_column_type_changes(
                    train_df, train_type_changes, train_date_col
                )
                st.session_state['train_df'] = train_df
                st.write("Train column types updated successfully!")
                st.write("**Updated Numeric Columns (Train)**:", train_numeric_cols)
                st.write("**Updated Categorical Columns (Train)**:", train_categorical_cols)
                if train_date_col:
                    st.write("**Date Column (Train)**:", train_date_col)

            # Data Exploration
            if st.button("Run Train Data Exploration"):
                explore_data(train_df, train_date_col, train_numeric_cols, train_categorical_cols, "train")

            # Data Preprocessing
            st.markdown("#### Train Data Preprocessing")
            train_outlier_method = st.selectbox("Handle Outliers (Train)", ['None', 'Remove', 'Replace with Median'], key="train_outliers")
            train_outlier_method = train_outlier_method.lower() if train_outlier_method != 'None' else None
            train_normalize = st.checkbox("Normalize Numeric Columns (Train)", key="train_normalize")
            if st.button("Preprocess Train Data"):
                processed_train = preprocess_data(
                    train_df, train_numeric_cols, train_categorical_cols, train_date_col, 
                    train_outlier_method, train_normalize, "train"
                )
                st.session_state['processed_train'] = processed_train
                st.write("Train preprocessing complete!")
                st.dataframe(processed_train.head())

                # Save processed train data
                save_option = st.selectbox("Save Processed Train Data As", ['CSV', 'Parquet'], key="train_save")
                if st.button("Save Processed Train Data"):
                    output_path = os.path.join("data", f"processed_train_{train_file.name.split('.')[0]}.{save_option.lower()}")
                    if save_option == 'CSV':
                        processed_train.to_csv(output_path, index=False)
                    else:
                        processed_train.to_parquet(output_path, index=False)
                    st.success(f"Processed train data saved to {output_path}")
                    os.system(f"dvc add {output_path}")
                    os.system(f"git add {output_path}.dvc")
                    os.system('git commit -m "Add processed train dataset to DVC"')

        # Test Data Processing
        if 'test_df' in st.session_state:
            test_df = st.session_state['test_df']
            st.subheader("Test Data Configuration")
            test_date_col = st.selectbox("Select Date Column for Test (optional)", ['None'] + list(test_df.columns), index=0, key="test_date")
            test_date_col = None if test_date_col == 'None' else test_date_col

            # Detect column types
            test_numeric_cols, test_categorical_cols = detect_column_types(test_df, test_date_col)
            st.write("**Detected Numeric Columns (Test)**:", test_numeric_cols)
            st.write("**Detected Categorical Columns (Test)**:", test_categorical_cols)

            # Allow column type changes
            st.markdown("#### Change Test Column Types (Optional)")
            test_type_changes = {}
            for col in test_df.columns:
                if col == test_date_col:
                    continue
                current_type = 'numeric' if col in test_numeric_cols else 'categorical' if col in test_categorical_cols else 'other'
                new_type = st.selectbox(
                    f"Type for {col} (Test, current: {current_type})",
                    ['Keep as is', 'Numeric', 'Categorical', 'Date'],
                    key=f"test_type_{col}"
                )
                if new_type != 'Keep as is':
                    test_type_changes[col] = new_type.lower()

            if st.button("Apply Test Column Type Changes"):
                test_df, test_numeric_cols, test_categorical_cols, test_date_col = apply_column_type_changes(
                    test_df, test_type_changes, test_date_col
                )
                st.session_state['test_df'] = test_df
                st.write("Test column types updated successfully!")
                st.write("**Updated Numeric Columns (Test)**:", test_numeric_cols)
                st.write("**Updated Categorical Columns (Test)**:", test_categorical_cols)
                if test_date_col:
                    st.write("**Date Column (Test)**:", test_date_col)

            # Data Exploration
            if st.button("Run Test Data Exploration"):
                explore_data(test_df, test_date_col, test_numeric_cols, test_categorical_cols, "test")

            # Data Preprocessing
            st.markdown("#### Test Data Preprocessing")
            test_outlier_method = st.selectbox("Handle Outliers (Test)", ['None', 'Remove', 'Replace with Median'], key="test_outliers")
            test_outlier_method = test_outlier_method.lower() if test_outlier_method != 'None' else None
            test_normalize = st.checkbox("Normalize Numeric Columns (Test)", key="test_normalize")
            if st.button("Preprocess Test Data"):
                processed_test = preprocess_data(
                    test_df, test_numeric_cols, test_categorical_cols, test_date_col, 
                    test_outlier_method, test_normalize, "test"
                )
                st.session_state['processed_test'] = processed_test
                st.write("Test preprocessing complete!")
                st.dataframe(processed_test.head())

                # Save processed test data
                save_option = st.selectbox("Save Processed Test Data As", ['CSV', 'Parquet'], key="test_save")
                if st.button("Save Processed Test Data"):
                    output_path = os.path.join("data", f"processed_test_{test_file.name.split('.')[0]}.{save_option.lower()}")
                    if save_option == 'CSV':
                        processed_test.to_csv(output_path, index=False)
                    else:
                        processed_test.to_parquet(output_path, index=False)
                    st.success(f"Processed test data saved to {output_path}")
                    os.system(f"dvc add {output_path}")
                    os.system(f"git add {output_path}.dvc")
                    os.system('git commit -m "Add processed test dataset to DVC"')

if __name__ == "__main__":
    main()
