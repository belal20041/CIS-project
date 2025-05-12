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
import uuid

# Custom CSS for visual appeal
st.markdown("""
<style>
    .main { background-color: #f9fafb; padding: 20px; }
    .stButton>button {
        background-color: #2E86C1; color: white; border-radius: 8px; padding: 8px 16px;
        border: none; transition: all 0.3s; }
    .stButton>button:hover { background-color: #1b6ca8; }
    .stSelectbox, .stCheckbox { background-color: white; border-radius: 8px; padding: 10px; }
    .card { background-color: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    h1, h2, h3 { color: #2E86C1; font-family: 'Roboto', sans-serif; }
    hr { border: 1px solid #e5e7eb; margin: 20px 0; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Milestone 1", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Collection & Preprocessing</h1>", unsafe_allow_html=True)

def load_data(file, dataset_type="train"):
    """Load dataset. Assigned to: Mohamed Samy (Standardization)"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        st.error("Unsupported format. Use CSV or Parquet.")
        return None
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Hoda Magdy: Fix inconsistencies
    return df

def detect_column_types(df, date_col=None):
    """Detect column types. Assigned to: Mohamed Samy"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = [
        col for col in df.columns
        if col != date_col and (df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() / len(df) < 0.05)
    ]
    if date_col:
        numeric_cols = [col for col in numeric_cols if col != date_col]
        categorical_cols = [col for col in categorical_cols if col != date_col]
    return numeric_cols, categorical_cols

def apply_column_type_changes(df, type_changes, date_col=None):
    """Apply type changes. Assigned to: Mohamed Samy"""
    df_modified = df.copy()
    new_numeric_cols, new_categorical_cols = [], []
    for col, new_type in type_changes.items():
        if col == date_col and new_type != 'date':
            st.warning(f"Cannot change date column {col} to {new_type}.")
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
        numeric_cols = [col for col in numeric_cols if col != date_col]
        categorical_cols = [col for col in categorical_cols if col != date_col]
    return df_modified, numeric_cols, categorical_cols, date_col

def explore_data(df, date_col=None, numeric_cols=None, categorical_cols=None, dataset_type="train"):
    """EDA: trends, seasonality, outliers. Assigned to: Belal Khamis"""
    with st.container():
        st.markdown(f"<div class='card'><h3>{dataset_type.capitalize()} Data Insights</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape**:", df.shape)
            st.write("**Missing Values**:", df.isna().sum().to_dict())
            st.write(f"**Duplicates**: {df.duplicated().sum()}")
        with col2:
            st.write("**Data Types**:")
            st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Type', 'index': 'Column'}))

        # Visualizations
        sns.set_style("whitegrid")
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                if 'sales' in df.columns:
                    st.markdown("**Sales Trends**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    df.groupby(date_col)['sales'].sum().plot(ax=ax)
                    ax.set_title("Sales Over Time")
                    st.pyplot(fig)

                    st.markdown("**Monthly Seasonality**")
                    df['month'] = df[date_col].dt.month
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x='month', y='sales', data=df, ax=ax)
                    ax.set_title("Sales by Month")
                    st.pyplot(fig)
            except:
                st.warning(f"Cannot process {date_col} for trends.")

        if 'sales' in df.columns and 'onpromotion' in df.columns:
            st.markdown("**Sales vs. Promotions**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(x='onpromotion', y='sales', data=df, ax=ax)
            ax.set_title("Sales vs. Promotions")
            st.pyplot(fig)

        if 'transferred' in df.columns:
            st.markdown("**Holiday Distribution**")
            st.dataframe(df['transferred'].value_counts())

        if numeric_cols:
            valid_numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df[col])]
            if valid_numeric_cols:
                st.markdown("**Outliers**")
                fig, axes = plt.subplots(1, len(valid_numeric_cols), figsize=(4*len(valid_numeric_cols), 4))
                axes = [axes] if len(valid_numeric_cols) == 1 else axes
                for i, col in enumerate(valid_numeric_cols):
                    sns.boxplot(x=df[col], ax=axes[i])
                    axes[i].set_title(f"{col}")
                plt.tight_layout()
                st.pyplot(fig)

                st.write("**Outlier Counts**:")
                for col in valid_numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    z_scores = zscore(df[col].dropna()) if df[col].std() > 1e-6 else []
                    outliers_zscore = df[abs(z_scores) > 3][col] if z_scores else pd.Series(dtype=df[col].dtype)
                    st.write(f"{col}: IQR Outliers = {len(outliers_iqr)}, Z-score Outliers = {len(outliers_zscore)}")
        st.markdown("</div>", unsafe_allow_html=True)

def preprocess_data(df, numeric_cols=None, categorical_cols=None, date_col=None, handle_outliers='remove', normalize=False, dataset_type="train"):
    """Preprocess data. Assigned to: Belal Khamis, Marwa Kotb, Mahmoud Sabry, Mohamed Samy, Hoda Magdy"""
    df_clean = df.copy()

    # Missing values. Assigned to: Belal Khamis
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif col in categorical_cols:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)

    # Duplicates. Assigned to: Marwa Kotb
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()

    # Outliers. Assigned to: Mahmoud Sabry
    if numeric_cols and handle_outliers:
        if handle_outliers == 'remove':
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
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

    # Time features. Assigned to: Mohamed Samy
    if date_col:
        try:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean['day'] = df_clean[date_col].dt.day
            df_clean['month'] = df_clean[date_col].dt.month
            df_clean['year'] = df_clean[date_col].dt.year
            df_clean['dayofweek'] = df_clean[date_col].dt.dayofweek
            df_clean['sin_month'] = np.sin(2 * np.pi * df_clean['month'] / 12)
        except:
            st.warning(f"Cannot process time features for {date_col}.")

    # Normalization. Assigned to: Mohamed Samy
    if normalize and numeric_cols:
        scaler = MinMaxScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    return df_clean, initial_rows - df_clean.shape[0]

def main():
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("ℹ️ Project Info"):
        st.write("**Objective**: Collect, explore, and preprocess historical sales data for analysis.")
        st.write("**Team**:")
        st.write("- Belal Khamis: Missing Values, EDA")
        st.write("- Marwa Kotb: Duplicates")
        st.write("- Mahmoud Sabry: Outliers")
        st.write("- Mohamed Samy: Standardization, Time Features, Normalization")
        st.write("- Hoda Magdy: Inconsistencies")

    # Tabs for Train and Test
    train_tab, test_tab = st.tabs(["Train Data", "Test Data"])

    with train_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        train_file = st.file_uploader("Upload Train Data", type=['csv', 'parquet'], key="train")
        if train_file:
            train_df = load_data(train_file, "train")
            if train_df is not None:
                st.session_state['train_df'] = train_df
                st.write("**Preview**:")
                st.dataframe(train_df.head(), height=150)

                # Configuration
                with st.form("train_config"):
                    train_date_col = st.selectbox("Date Column", ['None'] + list(train_df.columns), index=0)
                    train_date_col = None if train_date_col == 'None' else train_date_col
                    train_numeric_cols, train_categorical_cols = detect_column_types(train_df, train_date_col)

                    with st.expander("Adjust Column Types"):
                        train_type_changes = {}
                        for col in train_df.columns:
                            if col == train_date_col:
                                continue
                            current_type = 'numeric' if col in train_numeric_cols else 'categorical' if col in train_categorical_cols else 'other'
                            new_type = st.selectbox(
                                f"{col} (current: {current_type})",
                                ['Keep as is', 'Numeric', 'Categorical', 'Date'],
                                key=f"train_type_{col}_{uuid.uuid4()}"
                            )
                            if new_type != 'Keep as is':
                                train_type_changes[col] = new_type.lower()

                    st.form_submit_button("Apply Types", on_click=lambda: st.session_state.update({
                        'train_df': apply_column_type_changes(st.session_state['train_df'], train_type_changes, train_date_col)[0],
                        'train_numeric_cols': apply_column_type_changes(st.session_state['train_df'], train_type_changes, train_date_col)[1],
                        'train_categorical_cols': apply_column_type_changes(st.session_state['train_df'], train_type_changes, train_date_col)[2],
                        'train_date_col': apply_column_type_changes(st.session_state['train_df'], train_type_changes, train_date_col)[3]
                    }))

                # Exploration and Preprocessing
                with st.form("train_process"):
                    st.markdown("**Process Train Data**")
                    train_outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace with Median'], key="train_outliers")
                    train_outlier_method = train_outlier_method.lower() if train_outlier_method != 'None' else None
                    train_normalize = st.checkbox("Normalize Numerics")
                    save_format = st.selectbox("Save As", ['CSV', 'Parquet'], key="train_save")
                    submitted = st.form_submit_button("Explore & Preprocess")

                    if submitted:
                        train_df = st.session_state['train_df']
                        train_numeric_cols = st.session_state.get('train_numeric_cols', detect_column_types(train_df, train_date_col)[0])
                        train_categorical_cols = st.session_state.get('train_categorical_cols', detect_column_types(train_df, train_date_col)[1])
                        train_date_col = st.session_state.get('train_date_col', train_date_col)

                        # Exploration
                        explore_data(train_df, train_date_col, train_numeric_cols, train_categorical_cols, "train")

                        # Preprocessing
                        processed_train, duplicates_removed = preprocess_data(
                            train_df, train_numeric_cols, train_categorical_cols, train_date_col,
                            train_outlier_method, train_normalize, "train"
                        )
                        st.session_state['processed_train'] = processed_train
                        st.write(f"**Processed**: {duplicates_removed} duplicates removed, {processed_train.shape[0]} rows remain")
                        st.dataframe(processed_train.head(), height=150)

                        # Save
                        output_path = os.path.join("data", f"processed_train_{train_file.name.split('.')[0]}.{save_format.lower()}")
                        if save_format == 'CSV':
                            processed_train.to_csv(output_path, index=False)
                        else:
                            processed_train.to_parquet(output_path, index=False)
                        st.success(f"Saved to {output_path}")
                        os.system(f"dvc add {output_path}")
                        os.system(f"git add {output_path}.dvc")
                        os.system('git commit -m "Add processed train dataset to DVC"')
        st.markdown("</div>", unsafe_allow_html=True)

    with test_tab:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        test_file = st.file_uploader("Upload Test Data", type=['csv', 'parquet'], key="test")
        if test_file:
            test_df = load_data(test_file, "test")
            if test_df is not None:
                st.session_state['test_df'] = test_df
                st.write("**Preview**:")
                st.dataframe(test_df.head(), height=150)

                # Configuration
                with st.form("test_config"):
                    test_date_col = st.selectbox("Date Column", ['None'] + list(test_df.columns), index=0)
                    test_date_col = None if test_date_col == 'None' else test_date_col
                    test_numeric_cols, test_categorical_cols = detect_column_types(test_df, test_date_col)

                    with st.expander("Adjust Column Types"):
                        test_type_changes = {}
                        for col in test_df.columns:
                            if col == test_date_col:
                                continue
                            current_type = 'numeric' if col in test_numeric_cols else 'categorical' if col in test_categorical_cols else 'other'
                            new_type = st.selectbox(
                                f"{col} (current: {current_type})",
                                ['Keep as is', 'Numeric', 'Categorical', 'Date'],
                                key=f"test_type_{col}_{uuid.uuid4()}"
                            )
                            if new_type != 'Keep as is':
                                test_type_changes[col] = new_type.lower()

                    st.form_submit_button("Apply Types", on_click=lambda: st.session_state.update({
                        'test_df': apply_column_type_changes(st.session_state['test_df'], test_type_changes, test_date_col)[0],
                        'test_numeric_cols': apply_column_type_changes(st.session_state['test_df'], test_type_changes, test_date_col)[1],
                        'test_categorical_cols': apply_column_type_changes(st.session_state['test_df'], test_type_changes, test_date_col)[2],
                        'test_date_col': apply_column_type_changes(st.session_state['test_df'], test_type_changes, test_date_col)[3]
                    }))

                # Exploration and Preprocessing
                with st.form("test_process"):
                    st.markdown("**Process Test Data**")
                    test_outlier_method = st.selectbox("Outliers", ['None', 'Remove', 'Replace with Median'], key="test_outliers")
                    test_outlier_method = test_outlier_method.lower() if test_outlier_method != 'None' else None
                    test_normalize = st.checkbox("Normalize Numerics")
                    save_format = st.selectbox("Save As", ['CSV', 'Parquet'], key="test_save")
                    submitted = st.form_submit_button("Explore & Preprocess")

                    if submitted:
                        test_df = st.session_state['test_df']
                        test_numeric_cols = st.session_state.get('test_numeric_cols', detect_column_types(test_df, test_date_col)[0])
                        test_categorical_cols = st.session_state.get('test_categorical_cols', detect_column_types(test_df, test_date_col)[1])
                        test_date_col = st.session_state.get('test_date_col', test_date_col)

                        # Exploration
                        explore_data(test_df, test_date_col, test_numeric_cols, test_categorical_cols, "test")

                        # Preprocessing
                        processed_test, duplicates_removed = preprocess_data(
                            test_df, test_numeric_cols, test_categorical_cols, test_date_col,
                            test_outlier_method, test_normalize, "test"
                        )
                        st.session_state['processed_test'] = processed_test
                        st.write(f"**Processed**: {duplicates_removed} duplicates removed, {processed_test.shape[0]} rows remain")
                        st.dataframe(processed_test.head(), height=150)

                        # Save
                        output_path = os.path.join("data", f"processed_test_{test_file.name.split('.')[0]}.{save_format.lower()}")
                        if save_format == 'CSV':
                            processed_test.to_csv(output_path, index=False)
                        else:
                            processed_test.to_parquet(output_path, index=False)
                        st.success(f"Saved to {output_path}")
                        os.system(f"dvc add {output_path}")
                        os.system(f"git add {output_path}.dvc")
                        os.system('git commit -m "Add processed test dataset to DVC"')
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
